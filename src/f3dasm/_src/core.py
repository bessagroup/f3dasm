"""
This module contains the core blocks and protocols for the f3dasm package.
"""
#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Literal,
                    Optional, Protocol, Tuple)

# Third-party
import numpy as np
import pandas as pd
from pathos.helpers import mp

# Local
from .logger import logger
from .utils import number_of_overiterations, number_of_updates

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Alpha'
# =============================================================================
#
# =============================================================================


class Block(ABC):
    def arm(self, data: ExperimentData) -> None:
        self.data = data

    @abstractmethod
    def call(self, **kwargs) -> ExperimentData:
        pass

# =============================================================================


class Domain(Protocol):
    ...


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
    def __init__(self, domain: Domain, input_data: np.ndarray,
                 output_data: np.ndarray):
        ...

    @property
    def domain(self) -> Domain:
        ...

    @property
    def project_dir(self) -> Path:
        ...

    @property
    def index(self) -> pd.Index:
        ...

    @classmethod
    def from_sampling(cls, domain: Domain, sampler: Sampler,
                      n_samples: int, seed: int) -> ExperimentData:
        ...

    def access_file(self, operation: Callable) -> Callable:
        ...

    def get_open_job(self) -> Tuple[int, ExperimentSample]:
        ...

    def store_experimentsample(self,
                               experiment_sample: ExperimentSample, id: int):
        ...

    def from_file(self, project_dir: str) -> ExperimentData:
        ...

    def store(self) -> None:
        ...

    def mark(self, indices: int | Iterable[int],
             status: Literal['open', 'in_progress', 'finished', 'error']):
        ...

    def remove_lockfile(self) -> None:
        ...

    def sample(self, sampler: Sampler, **kwargs):
        ...

    def evaluate(self, data_generator: DataGenerator, mode:
                 str, output_names: Optional[List[str]] = None, **kwargs):
        ...

    def get_n_best_output(self, n_samples: int) -> ExperimentData:
        ...

    def to_numpy() -> Tuple[np.ndarray, np.ndarray]:
        ...

    def select(self, indices: int | slice | Iterable[int]) -> ExperimentData:
        ...

    def get_experiment_sample(self, id: int) -> ExperimentData:
        ...

    def remove_rows_bottom(self, n_rows: int):
        ...

    def add_experiments(self, experiment_sample: ExperimentData):
        ...

    def _overwrite_experiments(self, experiment_sample: ExperimentData,
                               indices: pd.Index, add_if_not_exist: bool):
        ...

    def _reset_index(self):
        ...

# =============================================================================


class Sampler(Block):
    pass

# =============================================================================


class DataGenerator(Block):
    """Base class for a data generator"""

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
            self._evaluate_cluster(**kwargs)
        else:
            raise ValueError(f"Invalid parallelization mode specified: {mode}")

        return self.data

    # =========================================================================

    def _evaluate_sequential(self, **kwargs):
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

            job_number, experiment_sample = self.data.get_open_job()
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

    def _evaluate_multiprocessing(self, nodes: int = mp.cpu_count(), **kwargs):
        options = []
        while True:
            job_number, experiment_sample = self.data.get_open_job()
            if job_number is None:
                break
            options.append(
                ({'experiment_sample': experiment_sample,
                  '_job_number': job_number, **kwargs},))

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

    def _evaluate_cluster(self, **kwargs):
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

        get_open_job = self.data.access_file(type(self.data).get_open_job)
        store_experiment_sample = self.data.access_file(
            type(self.data).store_experimentsample)
        mark = self.data.access_file(type(self.data).mark)

        while True:

            job_number, experiment_sample = get_open_job()
            if job_number is None:
                logger.debug("No Open jobs left!")
                break

            try:
                _experiment_sample = self._run(
                    experiment_sample, **kwargs)
                store_experiment_sample(experiment_sample=_experiment_sample,
                                        id=job_number)
            except Exception:
                # n = experiment_sample.job_number
                error_msg = f"Error in experiment_sample {job_number}: "
                error_traceback = traceback.format_exc()
                logger.error(f"{error_msg}\n{error_traceback}")
                mark(indices=job_number, status='error')
                continue

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
            self, experiment_sample: ExperimentSample,
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

        kwargs : dict
            The keyword arguments to pass to the pre_process, execute \
            and post_process

        Returns
        -------
        ExperimentSample
            Processed design with the response of the data generator \
            saved in the experiment_sample
        """
        self.experiment_sample = experiment_sample

        self.experiment_sample.mark('in_progress')

        self.execute(**kwargs)

        self.experiment_sample.mark('finished')

        return self.experiment_sample

# =============================================================================


class Optimizer:
    """
    Abstract class for optimization algorithms
    To implement a new optimizer, inherit from this class and implement the
    update_step method.

    Note
    ----
    The update_step method should have the following signature:

    '''
    def update_step(self)
    -> Tuple[np.ndarray, np.ndarray]:
    '''

    The method should return a tuple containing the new samples and the
    corresponding objective values. The new samples should be a numpy array.

    If the optimizer requires gradients, set the require_gradients attribute to
    True. This will ensure that the data_generator will calculate the gradients
    of the objective function from the DataGenerator.dfdx method.

    Hyperparameters can be set in the __init__ method of the child class.
    There are two hyperparameters that have special meaning:
    - population: the number of individuals in the population
    - seed: the seed of the random number generator

    You can create extra methods in your child class as you please, however
    it is advised not to create private methods (methods starting with an
    underscore) as these might be used in the base class.
    """
    type: ClassVar[str] = 'any'
    require_gradients: ClassVar[bool] = False

#                                                            Private Properties
# =============================================================================

    @property
    def _seed(self) -> int:
        """
        Property to return the seed of the optimizer

        Returns
        -------
        int | None
            Seed of the optimizer

        Note
        ----
        If the seed is not set, the property will return None
        This is done to prevent errors when the seed is not an available
        attribute in a custom optimizer class.
        """
        return self.seed if hasattr(self, 'seed') else None

    @property
    def _population(self) -> int:
        """
        Property to return the population size of the optimizer

        Returns
        -------
        int
            Number of individuals in the population

        Note
        ----
        If the population is not set, the property will return 1
        This is done to prevent errors when the population size is not an
        available attribute in a custom optimizer class.
        """
        return self.population if hasattr(self, 'population') else 1

#                                                                Public Methods
# =============================================================================

    @abstractmethod
    def update_step(self) -> ExperimentData:
        """Update step of the optimizer. Needs to be implemented
         by the child class

        Returns
        -------
        ExperimentData
            ExperimentData object containing the new samples

        Raises
        ------
        NotImplementedError
            Raises when the method is not implemented by the child class

        Note
        ----
        You can access the data attribute of the optimizer to get the
        available data points. The data attribute is an
        f3dasm.ExperimentData object.
        """
        raise NotImplementedError(
            "You should implement an update step for your optimizer!")

    def arm(self, data: ExperimentData, data_generator: DataGenerator):
        """
        Initialize the optimizer with the given data and data generator

        Parameters
        ----------
        data : ExperimentData
            Data to initialize the optimizer
        data_generator : DataGenerator
            Data generator to calculate the objective value
        """
        self.data = data
        self.data_generator = data_generator

    def call(self, iterations: int, last_index: int,
             kwargs: Optional[Dict[str, Any]] = None,
             x0_selection: Literal['best', 'random',
                                   'last',
                                   'new'] | ExperimentData = 'best',
             sampler: Optional[Sampler | str] = 'random',
             overwrite: bool = False,
             callback: Optional[Callable] = None,
             ) -> ExperimentData:

        return self._iterate(
            iterations=iterations, kwargs=kwargs,
            x0_selection=x0_selection,
            sampler=sampler,
            overwrite=overwrite,
            callback=callback,
            last_index=last_index)

    def _iterate(self, iterations: int, kwargs: Dict[str, Any],
                 x0_selection: str, sampler: Sampler, overwrite: bool,
                 callback: Callable, last_index: int):
        """Internal represenation of the iteration process

        Parameters
        ----------
        optimizer : Optimizer
            Optimizer object
        data_generator : DataGenerator
            DataGenerator object
        iterations : int
            number of iterations
        kwargs : Dict[str, Any]
            any additional keyword arguments that will be passed to
            the DataGenerator
        x0_selection : str | ExperimentData
            How to select the initial design.
            The following x0_selections are available:

            * 'best': Select the best designs from the current experimentdata
            * 'random': Select random designs from the current experimentdata
            * 'last': Select the last designs from the current experimentdata
            * 'new': Create new random designs from the current experimentdata

            If the x0_selection is 'new', new designs are sampled with the
            sampler provided. The number of designs selected is equal to the
            population size of the optimizer.

            If an ExperimentData object is passed as x0_selection,
            the optimizer will use the input_data and output_data from this
            object as initial samples.

        sampler: Sampler
            If x0_selection = 'new', the sampler to use
        overwrite: bool
            If True, the optimizer will overwrite the current data.
        callback : Callable
            A callback function that is called after every iteration. It has
            the following signature:

                    ``callback(intermediate_result: ExperimentData)``

            where the first argument is a parameter containing an
            `ExperimentData` object with the current iterate(s).

        Raises
        ------
        ValueError
            Raised when invalid x0_selection is specified
        """
        if len(self.data) < self._population:
            raise ValueError(
                f'There are {len(self.data)} datapoints available, \
                        need {self._population} for initial \
                            population!'
            )

        for _ in range(number_of_updates(
                iterations,
                population=self._population)):
            new_samples = self.update_step()

            # If new_samples is a tuple of input_data and output_data
            if isinstance(new_samples, tuple):
                new_samples = type(self.data)(
                    domain=self.data.domain,
                    input_data=new_samples[0],
                    output_data=new_samples[1],
                )

            # If applicable, evaluate the new designs:
            new_samples.evaluate(data_generator=self.data_generator,
                                 mode='sequential', **kwargs)

            if callback is not None:
                callback(new_samples)

            if overwrite:
                _indices = new_samples.index + last_index + 1
                self.data._overwrite_experiments(experiment_sample=new_samples,
                                                 indices=_indices,
                                                 add_if_not_exist=True)

            else:
                self.data.add_experiments(new_samples)

        if not overwrite:
            # Remove overiterations
            self.data.remove_rows_bottom(number_of_overiterations(
                iterations,
                population=self._population))

        return self.data.select(self.data.index[self._population:])
