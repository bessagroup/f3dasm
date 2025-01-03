"""
Interface class for data generators
"""

#                                                                       Modules
# =============================================================================


from __future__ import annotations

# Standard
import inspect
import traceback
from abc import abstractmethod
from pathlib import Path
from typing import (Any, Callable, Dict, Iterable, List, Literal, NamedTuple,
                    Optional, Protocol, Tuple, Type)

# Third-party
import numpy as np
from pathos.helpers import mp

# Local
from ..design.domain import Domain
from ..logger import logger

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Alpha"
# =============================================================================
#
# =============================================================================


class ExperimentSample(Protocol):
    def mark(self,
             status: Literal['open', 'in_progress', 'finished', 'error']):
        ...

    @property
    def input_data(self) -> Dict[str, Any]:
        ...

    @property
    def output_data(self) -> Dict[str, Any]:
        ...


class ExperimentData(Protocol):

    @property
    def project_dir(self) -> Path:
        ...

    def _access_open_job_data(self) -> Tuple[int, ExperimentSample]:
        ...

    def store_experimentsample(self,
                               experiment_sample: ExperimentSample, id: int):
        ...

    def _set_error(self, job_number: int) -> None:
        ...

    def _get_open_job_data(self) -> ExperimentSample:
        ...

    def _write_experiment_sample(self,
                                 experiment_sample: ExperimentSample) -> None:
        ...

    def _write_error(self, job_number: int) -> None:
        ...

    def from_file(self, project_dir: str) -> ExperimentData:
        ...

    def store(self) -> None:
        ...

    def select(self, indices: List[int]) -> ExperimentData:
        ...

    def overwrite_disk(
            self, indices: List[int], input_data: np.ndarray,
            output_data: np.ndarray, jobs: np.ndarray, domain: Domain,
            add_if_not_exist: bool) -> None:
        ...

    def mark(self, indices: int | Iterable[int],
             status: Literal['open', 'in_progress', 'finished', 'error']):
        ...

    def remove_lockfile(self) -> None:
        ...


# =============================================================================


class DataGenerator:
    """Base class for a data generator"""

    def init(self, data: ExperimentData):
        self.data = data

    def call(self, mode: str = 'sequential', **kwargs
             ) -> ExperimentData:
        """
        Evaluate the data generator.

        Parameters
        ----------
        mode : str, optional
            The mode of evaluation, by default 'sequential'
        kwargs : dict
            The keyword arguments to pass to the pre_process, execute and
            post_process

        Returns
        -------
        ExperimentData
            The processed data
        """
        if mode == 'sequential':
            self._evaluate_sequential(**kwargs)
        elif mode == 'parallel':
            self._evaluate_multiprocessing(**kwargs)
        elif mode.lower() == "cluster":
            return self._evaluate_cluster(**kwargs)
        elif mode.lower() == "cluster_parallel":
            return self._evaluate_cluster_parallel(**kwargs)
        else:
            raise ValueError(f"Invalid parallelization mode specified: {mode}")

        return self.data

    def _evaluate_sequential(self, **kwargs):
        """Run the operation sequentially

        Parameters
        ----------
        operation : ExperimentSampleCallable
            function execution for every entry in the ExperimentData object
        kwargs : dict
            Any keyword arguments that need to be supplied to the function

        Raises
        ------
        NoOpenJobsError
            Raised when there are no open jobs left
        """
        while True:

            job_number, experiment_sample = self.data._access_open_job_data()
            logger.debug(
                f"Accessed experiment_sample \
                        {job_number}")
            if job_number is None:
                logger.debug("No Open Jobs left")
                break

            try:

                # If kwargs is empty dict
                if not kwargs:
                    logger.debug(
                        f"Running experiment_sample "
                        f"{job_number}")
                else:
                    logger.debug(
                        f"Running experiment_sample "
                        f"{job_number} with kwargs {kwargs}")

                _experiment_sample = self._run(
                    experiment_sample, **kwargs)  # no *args!
                self.data.store_experimentsample(
                    experiment_sample=_experiment_sample,
                    id=job_number)
            except Exception as e:
                error_msg = f"Error in experiment_sample \
                     {job_number}: {e}"
                error_traceback = traceback.format_exc()
                logger.error(f"{error_msg}\n{error_traceback}")
                self.data.mark(indices=job_number, status='error')

    def _evaluate_multiprocessing(self, **kwargs):
        """Run the operation on multiple cores

        Parameters
        ----------
        operation : ExperimentSampleCallable
            function execution for every entry in the ExperimentData object
        kwargs : dict
            Any keyword arguments that need to be supplied to the function

        Raises
        ------
        NoOpenJobsError
            Raised when there are no open jobs left
        """
        # Get all the jobs
        options = []
        while True:
            job_number, experiment_sample = self.data._access_open_job_data()
            options.append(
                ({'experiment_sample': experiment_sample,
                  '_job_number': job_number, **kwargs},))

            if job_number is None:
                break

        def f(options: Dict[str, Any]) -> Tuple[int, ExperimentSample, int]:
            try:

                logger.debug(
                    f"Running experiment_sample "
                    f"{options['_job_number']}")

                # no *args!
                return (options['_job_number'], self._run(**options), 0)

            except Exception as e:
                error_msg = f"Error in experiment_sample \
                     {options['_job_number']}: {e}"
                error_traceback = traceback.format_exc()
                logger.error(f"{error_msg}\n{error_traceback}")
                return (options['_job_number'],
                        options['experiment_sample'], 1)

        with mp.Pool() as pool:
            # maybe implement pool.starmap_async ?
            _experiment_samples: List[
                Tuple[int, ExperimentSample, int]] = pool.starmap(f, options)

        for job_number, _experiment_sample, exit_code in _experiment_samples:
            if exit_code == 0:
                self.data.store_experimentsample(
                    experiment_sample=_experiment_sample,
                    id=job_number)
            else:
                self.data.mark(indices=job_number, status='error')

    def _evaluate_cluster(self, data_generator: DataGenerator, **kwargs):
        """Run the operation on the cluster

        Parameters
        ----------
        operation : ExperimentSampleCallable
            function execution for every entry in the ExperimentData object
        kwargs : dict
            Any keyword arguments that need to be supplied to the function

        Raises
        ------
        NoOpenJobsError
            Raised when there are no open jobs left
        """
        # Retrieve the updated experimentdata object from disc
        try:
            self.data = self.data.from_file(self.data.project_dir)
        except FileNotFoundError:  # If not found, store current
            self.data.store()

        while True:

            job_number, experiment_sample = self.data._get_open_job_data()
            if job_number is None:
                logger.debug("No Open jobs left!")
                break

            try:
                _experiment_sample = data_generator._run(
                    experiment_sample, **kwargs)
                self.data._write_experiment_sample(_experiment_sample)
            except Exception:
                # n = experiment_sample.job_number
                error_msg = f"Error in experiment_sample {job_number}: "
                error_traceback = traceback.format_exc()
                logger.error(f"{error_msg}\n{error_traceback}")
                self.data._write_error(job_number)
                continue

        self.data = self.data.from_file(self.data.project_dir)

        # Remove the lockfile from disk
        self.data.remove_lockfile()

    def _evaluate_cluster_parallel(
            self, data_generator: DataGenerator, **kwargs):
        """Run the operation on the cluster and parallelize it over cores

        Parameters
        ----------
        operation : ExperimentSampleCallable
            function execution for every entry in the ExperimentData object
        kwargs : dict
            Any keyword arguments that need to be supplied to the function

        Raises
        ------
        NoOpenJobsError
            Raised when there are no open jobs left
        """
        # Retrieve the updated experimentdata object from disc
        try:
            self.data = self.data.from_file(self.data.project_dir)
        except FileNotFoundError:  # If not found, store current
            self.data.store()

        no_jobs = False

        while True:
            es_list = []
            for core in range(mp.cpu_count()):
                job_number, experiment_sample = self.data._get_open_job_data()

                if job_number is None:
                    logger.debug("No Open jobs left!")
                    no_jobs = True
                    break

                es_list.append((job_number, self.data._get_open_job_data()))

            d = self.data.select([e[0] for e in es_list])

            # TODO: fix this; probably not working!
            self._evaluate_multiprocessing(**kwargs)

            # TODO access resource first!
            self.data.overwrite_disk(
                indices=d.index, input_data=d._input_data,
                output_data=d._output_data, jobs=d._jobs,
                domain=d.domain, add_if_not_exist=False)

            if no_jobs:
                break

        self.data = self.data.from_file(self.data.project_dir)
        # Remove the lockfile from disk
        self.data.remove_lockfile()

    # =========================================================================

    @abstractmethod
    def execute(self, **kwargs) -> None:
        """Interface function that handles the execution of the data generator

        Raises
        ------
        NotImplementedError
            If the function is not implemented by the user

        Note
        ----
        The experiment_sample is cached inside the data generator. This
        allows the user to access the experiment_sample in
        the execute and function as a class variable called
        self.experiment_sample.
        """
        ...

    def _run(
            self, experiment_sample: ExperimentSample | np.ndarray,
            domain: Optional[Domain] = None,
            **kwargs) -> ExperimentSample:
        """
        Run the data generator.

        The function also caches the experiment_sample in the data generator.
        This allows the user to access the experiment_sample in the
        execute function as a class variable
        called self.experiment_sample.

        Parameters
        ----------
        ExperimentSample : ExperimentSample
            The design to run the data generator on
        domain : Domain, optional
            The domain of the data generator, by default None

        kwargs : dict
            The keyword arguments to pass to the pre_process, execute \
            and post_process

        Returns
        -------
        ExperimentSample
            Processed design with the response of the data generator \
            saved in the experiment_sample
        """
        # Cache the design
        # self.experiment_sample: ExperimentSample = _experimentsample_factory(
        #     experiment_sample=experiment_sample, domain=domain)

        self.experiment_sample = experiment_sample

        self.execute(**kwargs)

        self.experiment_sample.mark('finished')

        return self.experiment_sample


def convert_function(f: Callable,
                     output: Optional[List[str]] = None,
                     kwargs: Optional[Dict[str, Any]] = None,
                     to_disk: Optional[List[str]] = None) -> DataGenerator:
    """
    Converts a given function `f` into a `DataGenerator` object.

    Parameters
    ----------
    f : Callable
        The function to be converted.
    output : Optional[List[str]], optional
        A list of names for the return values of the function.
        Defaults to None.
    kwargs : Optional[Dict[str, Any]], optional
        Additional keyword arguments passed to the function. Defaults to None.
    to_disk : Optional[List[str]], optional
        The list of output names where the value needs to be stored on disk.
        Defaults to None.

    Returns
    -------
    DataGenerator
        A converted `DataGenerator` object.

    Notes
    -----

    The function `f` can have any number of arguments and any number of returns
    as long as they are consistent with the `input` and `output` arguments that
    are given to this function.
    """
    signature = inspect.signature(f)
    input = list(signature.parameters)
    kwargs = kwargs if kwargs is not None else {}
    to_disk = to_disk if to_disk is not None else []
    output = output if output is not None else []

    class TempDataGenerator(DataGenerator):
        def execute(self, **_kwargs) -> None:
            _input = {input_name:
                      self.experiment_sample.input_data.get(input_name)
                      for input_name in input if input_name not in kwargs}
            _output = f(**_input, **kwargs)

            # check if output is empty
            if output is None:
                return

            if len(output) == 1:
                _output = (_output,)

            for name, value in zip(output, _output):
                if name in to_disk:
                    self.experiment_sample.store(name=name,
                                                 object=value,
                                                 to_disk=True)
                else:
                    self.experiment_sample.store(name=name,
                                                 object=value,
                                                 to_disk=False)

    return TempDataGenerator()

# =============================================================================


class BuiltinDataGenerator(NamedTuple):
    base_class: Type[DataGenerator]
    options: dict

    def init(self, domain: Domain) -> DataGenerator:
        return self.base_class(domain=domain, **self.options)
