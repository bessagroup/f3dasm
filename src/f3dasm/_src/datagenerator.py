"""
This module contains the core blocks and protocols for the f3dasm package.
"""
#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
import inspect
import traceback
from abc import abstractmethod
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple, Type

# Third-party
from filelock import FileLock
from pathos.helpers import mp

# Local
from ._io import EXPERIMENTDATA_SUBFOLDER, LOCK_FILENAME, MAX_TRIES
from .core import Block
from .experimentdata import ExperimentData
from .experimentsample import ExperimentSample
from .logger import logger
from .mpi_utils import (
    mpi_get_open_job,
    mpi_lock_manager,
    mpi_store_experiment_sample,
    mpi_terminate_worker,
)

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Alpha'
# =============================================================================
#
# =============================================================================


class DataGenerator(Block):
    """Base class for a data generator"""

    def call(self, data: ExperimentData | str,
             mode: str = 'sequential', pass_id: bool = False, **kwargs
             ) -> ExperimentData:
        """
        Evaluate the data generator.

        Parameters
        ----------
        data : ExperimentData | str
            The experiment data to process.
        mode : str, optional
            The mode of evaluation, by default 'sequential'
        pass_id : bool, optional
            Whether to pass the id to the execute function, by default False
        **kwargs : dict
            The keyword arguments to pass to execute function

        Returns
        -------
        ExperimentData
            The processed data

        Raises
        ------
        ValueError
            If an invalid mode is specified

        Notes
        -----
        The mode can be one of the following:
            - 'sequential': Run the data generator sequentially
            - 'parallel': Run the data generator in parallel
            - 'cluster': Run the data generator on a cluster
            - 'mpi': Run the data generator using MPI

        The 'pass_id' parameter is used to pass the id of the experiment sample
        to the execute function. This is useful when the execute function
        requires the id of the experiment sample to run. By default, this is
        set to False. The id is passed through the 'id' keyword argument.
        """
        data = data._copy(in_place=False, deep=True)

        if mode == 'sequential':
            return self._evaluate_sequential(data=data,
                                             pass_id=pass_id, ** kwargs)
        elif mode == 'parallel':
            return self._evaluate_multiprocessing(data=data,
                                                  pass_id=pass_id, **kwargs)
        elif mode.lower() == "cluster":
            return self._evaluate_cluster(data=data,
                                          pass_id=pass_id, **kwargs)
        elif mode.lower() == "mpi":
            return self._evaluate_mpi(data=data,
                                      pass_id=pass_id, **kwargs)
        else:
            raise ValueError(f"Invalid parallelization mode specified: {mode}")

    # =========================================================================

    def _evaluate_sequential(self, data: ExperimentData, pass_id: bool,
                             **kwargs) -> ExperimentData:
        """Run the operation sequentially

        Parameters
        ----------
        kwargs : dict
            Any keyword arguments that need to be supplied to the function

        Raises
        ------
        NoOpenJobsError
            Raised when there are no open jobs left
        """

        while True:
            job_number, experiment_sample = data.get_open_job()
            if job_number is None:
                logger.debug("No Open jobs left!")
                break

            try:
                if pass_id:
                    experiment_sample: ExperimentSample = self.execute(
                        experiment_sample=experiment_sample, id=job_number,
                        **kwargs)
                else:
                    experiment_sample: ExperimentSample = self.execute(
                        experiment_sample=experiment_sample, **kwargs)

                experiment_sample.store_experimentsample_references(
                    idx=job_number)

                experiment_sample.mark('finished')

            except Exception:
                error_msg = f"Error in experiment_sample {job_number}: "
                error_traceback = traceback.format_exc()
                logger.error(f"{error_msg}\n{error_traceback}")
                experiment_sample.mark('error')
                continue

            finally:
                data.store_experimentsample(
                    idx=job_number, experiment_sample=experiment_sample,
                )
        return data

    def _evaluate_multiprocessing(
        self, data: ExperimentData, pass_id: bool,
            nodes: int = mp.cpu_count(), **kwargs) -> ExperimentData:
        options = []

        while True:
            job_number, experiment_sample = data.get_open_job()
            if job_number is None:
                break

            if pass_id:
                options.append(
                    ({'experiment_sample': experiment_sample,
                      '_job_number': job_number, 'id': job_number, **kwargs},))
            else:
                options.append(
                    ({'experiment_sample': experiment_sample,
                      '_job_number': job_number, **kwargs},))

        def f(options: Dict[str, Any]) -> Tuple[int, ExperimentSample, int]:
            job_number = options.pop('_job_number')
            try:

                logger.debug(
                    f"Running experiment_sample "
                    f"{job_number}")

                experiment_sample: ExperimentSample = self.execute(**options)
                experiment_sample.store_experimentsample_references(
                    idx=job_number)
                # exit_code = 0
                experiment_sample.mark('finished')

            except Exception:
                error_msg = f"Error in experiment_sample {job_number}: "
                error_traceback = traceback.format_exc()
                logger.error(f"{error_msg}\n{error_traceback}")
                experiment_sample.mark('error')
                # exit_code = 1

            finally:
                return (job_number, experiment_sample)

        with mp.Pool() as pool:
            # maybe implement pool.starmap_async ?
            _experiment_samples: List[
                Tuple[int, ExperimentSample, int]] = pool.starmap(f, options)

        for job_number, experiment_sample in _experiment_samples:
            data.store_experimentsample(
                experiment_sample=experiment_sample,
                idx=job_number)

        return data

    def _evaluate_cluster(
        self, data: ExperimentData, pass_id: bool,
            wait_for_creation: bool = False,
            max_tries: int = MAX_TRIES, **kwargs
    ) -> None:

        # Creat lockfile
        lockfile = FileLock(
            (data.project_dir / EXPERIMENTDATA_SUBFOLDER / LOCK_FILENAME
             ).with_suffix('.lock'))

        cluster_get_open_job = partial(
            get_open_job, experiment_data_type=type(data),
            project_dir=data.project_dir,
            wait_for_creation=wait_for_creation, max_tries=max_tries,
            lockfile=lockfile)
        cluster_store_experiment_sample = partial(
            store_experiment_sample, experiment_data_type=type(data),
            project_dir=data.project_dir,
            wait_for_creation=wait_for_creation, max_tries=max_tries,
            lockfile=lockfile)

        while True:
            job_number, experiment_sample = cluster_get_open_job()
            if job_number is None:
                logger.debug("No Open jobs left!")
                break

            try:
                if pass_id:
                    experiment_sample: ExperimentSample = self.execute(
                        experiment_sample=experiment_sample, id=job_number,
                        **kwargs)
                else:
                    experiment_sample: ExperimentSample = self.execute(
                        experiment_sample=experiment_sample, **kwargs)

                experiment_sample.store_experimentsample_references(
                    idx=job_number)

                experiment_sample.mark('finished')

            except Exception:
                error_msg = f"Error in experiment_sample {job_number}: "
                error_traceback = traceback.format_exc()
                logger.error(f"{error_msg}\n{error_traceback}")
                experiment_sample.mark('error')
                continue

            finally:
                cluster_store_experiment_sample(
                    idx=job_number, experiment_sample=experiment_sample)

        # Remove lockfile
        (data.project_dir / EXPERIMENTDATA_SUBFOLDER / LOCK_FILENAME
         ).with_suffix('.lock').unlink(missing_ok=True)

    def _evaluate_mpi(
        self, comm, data: ExperimentData, pass_id: bool,
        wait_for_creation: bool = False,
        max_tries: int = MAX_TRIES, **kwargs
    ) -> None:
        rank = comm.Get_rank()
        size = comm.Get_size()

        if rank == 0:
            mpi_lock_manager(comm=comm, size=size)
        else:
            mpi_worker(comm=comm, data=data, execute_fn=self.execute,
                       pass_id=pass_id,
                       wait_for_creation=wait_for_creation,
                       max_tries=max_tries,
                       **kwargs)

    # =========================================================================

    @abstractmethod
    def execute(self, experiment_sample: ExperimentSample,
                **kwargs) -> ExperimentSample:
        """Interface function that handles the execution of the data generator

        Parameters
        ----------
        experiment_sample : ExperimentSample
            The experiment sample to run the data generator on
        kwargs : dict
            The optional keyword arguments to pass to the function

        Returns
        -------
        ExperimentSample
            The experiment sample with the response of the data generator
            saved in the experiment_sample

        Raises
        ------
        NotImplementedError
            If the function is not implemented by the user
        """
        ...

# =============================================================================


def get_open_job(experiment_data_type: Type[ExperimentData],
                 project_dir: Path, lockfile: FileLock,
                 wait_for_creation: bool, max_tries: int,
                 ) -> Tuple[int, ExperimentSample]:

    with lockfile:
        data: ExperimentData = experiment_data_type.from_file(
            project_dir=project_dir, wait_for_creation=wait_for_creation,
            max_tries=max_tries)

        idx, es = data.get_open_job()

        data.store(project_dir)

    return idx, es


def store_experiment_sample(
    experiment_data_type: Type[ExperimentData],
        project_dir: Path, lockfile: FileLock, wait_for_creation: bool,
        max_tries: int, idx: int, experiment_sample: ExperimentSample) -> None:

    with lockfile:
        data: ExperimentData = experiment_data_type.from_file(
            project_dir=project_dir, wait_for_creation=wait_for_creation,
            max_tries=max_tries)

        data.store_experimentsample(experiment_sample=experiment_sample,
                                    idx=idx)
        data.store(project_dir)


def mpi_worker(
    comm, data: ExperimentData,
        execute_fn: Callable,
        pass_id: bool,
        wait_for_creation: bool = False,
        max_tries: int = MAX_TRIES, **kwargs
) -> None:

    cluster_get_open_job = partial(
        mpi_get_open_job, experiment_data_type=type(data),
        project_dir=data.project_dir,
        wait_for_creation=wait_for_creation, max_tries=max_tries,
        comm=comm)
    cluster_store_experiment_sample = partial(
        mpi_store_experiment_sample, experiment_data_type=type(data),
        project_dir=data.project_dir,
        wait_for_creation=wait_for_creation, max_tries=max_tries,
        comm=comm)

    while True:
        job_number, experiment_sample = cluster_get_open_job()
        if job_number is None:
            logger.debug("No Open jobs left!")
            break

        try:
            if pass_id:
                experiment_sample: ExperimentSample = execute_fn(
                    experiment_sample=experiment_sample, id=job_number,
                    **kwargs)
            else:
                experiment_sample: ExperimentSample = execute_fn(
                    experiment_sample=experiment_sample, **kwargs)

            experiment_sample.store_experimentsample_references(
                idx=job_number)

            experiment_sample.mark('finished')

        except Exception:
            error_msg = f"Error in experiment_sample {job_number}: "
            error_traceback = traceback.format_exc()
            logger.error(f"{error_msg}\n{error_traceback}")
            experiment_sample.mark('error')
            continue

        finally:
            cluster_store_experiment_sample(
                idx=job_number, experiment_sample=experiment_sample)

    mpi_terminate_worker(comm)


def datagenerator(output_names: Iterable[str]
                  ) -> Callable[[Callable[..., Any]], DataGenerator]:
    """
    Decorator to convert a function into a `DataGenerator` subclass.

    The decorated function should take named arguments matching the keys in
    the Domain and return one or multiple output values. These values will be
    stored in the `ExperimentData` under the names specified in `output_names`.

    Parameters
    ----------
    output_names : Iterable[str]
        A list of names for the returned values of the decorated function.
        The function's return values will be stored in the `ExperimentSample`
        object under these names.

    Returns
    -------
    Callable[[Callable[..., Any]], DataGenerator]
        A decorator that transforms a function into a `DataGenerator` subclass.

    Raises
    ------
    ValueError
        If `output_names` is not provided.

    Examples
    --------
    >>> @datagenerator(output_names=['y'])
    ... def my_function(x0: float, x1: float, x2: float) -> float:
    ...     return x0**2 + x1**2 + x2**2
    ...
    >>> experiment_data = ExperimentData(domaind=domain)
    >>> experiment_data = my_function.call(experiment_data)
    """

    if not output_names:
        raise ValueError(
            "If you provide a function as a data generator, you must "
            "provide the names of the return arguments with the `output_names`"
            "attribute."
        )

    # If the output names is a single string, convert it to a list
    if isinstance(output_names, str):
        output_names = [output_names]

    def decorator(f: Callable[..., Any]) -> DataGenerator:
        signature = inspect.signature(f)

        class FunctionDataGenerator(DataGenerator):
            """
            Auto-generated DataGenerator subclass from a decorated function.
            """

            def execute(self, experiment_sample: ExperimentSample,
                        **kwargs: Any) -> ExperimentSample:
                """
                Executes the data generation process by calling the decorated
                function.

                Parameters
                ----------
                experiment_sample : ExperimentSample
                    The experiment sample containing input data.
                **kwargs : dict
                    Additional keyword arguments to override values in
                    `experiment_sample.input_data`.

                Returns
                -------
                ExperimentSample
                    The modified `ExperimentSample` instance with new stored
                    outputs.
                """
                _input = {name: val.default for name, val in
                          signature.parameters.items()
                          if val.default is not inspect.Parameter.empty
                          }

                _input.update(kwargs)

                _input.update({name: experiment_sample.input_data[name] for
                               name in experiment_sample.input_data
                               if name in signature.parameters})

                # Call the function
                _output = f(**_input)

                # Ensure the output is iterable
                if not isinstance(_output, tuple):
                    _output = (_output,)

                # Store outputs in the experiment sample
                for name, value in zip(output_names, _output):
                    experiment_sample.store(name=name, object=value)

                return experiment_sample

        data_generator = FunctionDataGenerator()
        data_generator.output_names = output_names
        data_generator.f = f
        return data_generator

    return decorator
