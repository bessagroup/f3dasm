"""
The ExperimentData object is the main object used to store implementations of a design-of-experiments,
keep track of results, perform optimization and extract data for machine learning purposes.
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
import json
import os
import sys
import traceback
from copy import deepcopy
from functools import wraps
from io import TextIOWrapper
from pathlib import Path
from time import sleep

if sys.version_info < (3, 8):  # NOQA
    from typing_extensions import Protocol  # NOQA
else:
    from typing import Protocol

from typing import (Any, Callable, Dict, Iterable, Iterator, List, Optional,
                    Tuple, Type, Union)

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from filelock import FileLock
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig
from pathos.helpers import mp

# Local
from ..logger import logger
from ._data import _Data
from ._jobqueue import NoOpenJobsError, Status, _JobQueue
from .domain import Domain
from .experimentsample import ExperimentSample
from .parameter import Parameter

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

DataTypes = Union[pd.DataFrame, np.ndarray, Path, str, _Data]


class _OptimizerParameters(Protocol):
    maxiter: int
    population: int


class _Optimizer(Protocol):
    hyperparameters: _OptimizerParameters
    type: str

    def _callback(self, xk: np.ndarray) -> None:
        ...

    def run_algorithm(self, iterations: int, data_generator: _DataGenerator):
        ...

    def _check_number_of_datapoints(self) -> None:
        ...

    def update_step(self, data_generator: _DataGenerator) -> ExperimentData:
        ...

    def _construct_model(self, data_generator: _DataGenerator) -> None:
        ...

    def set_x0(self, experiment_data: ExperimentData) -> None:
        ...

    def set_data(self, data: ExperimentData) -> None:
        ...

    def reset(self) -> None:
        ...


class _DataGenerator(Protocol):
    def run(self, experiment_sample: ExperimentSample) -> ExperimentSample:
        ...


class _Sampler(Protocol):
    """Protocol class for sampling methods."""
    def get_samples(numsamples: int) -> ExperimentData:
        ...

    @classmethod
    def from_yaml(cls, domain_config: DictConfig, sampler_config: DictConfig) -> '_Sampler':
        """Create a sampler from a yaml configuration"""

        args = {**sampler_config, 'domain': None}
        sampler: _Sampler = instantiate(args)
        sampler.domain = Domain.from_yaml(domain_config)
        return sampler


class _ExperimentSampleCallable(Protocol):
    def __call__(experiment_sample: ExperimentSample, **kwargs) -> ExperimentSample:
        ...


class ExperimentData:
    """
    A class that contains data for experiments.
    """

    def __init__(self, domain: Optional[Domain] = None, input_data: Optional[DataTypes] = None,
                 output_data: Optional[DataTypes] = None, jobs: Optional[Path | str] = None,
                 filename: Optional[str] = 'experimentdata'):
        """
        Initializes an instance of ExperimentData.

        Parameters
        ----------
        domain : Domain, optional
            The domain of the experiment, by default None
        input_data : DataTypes, optional
            The input data of the experiment, by default None
        output_data : DataTypes, optional
            The output data of the experiment, by default None
        jobs : Path | str, optional
            The path to the jobs file, by default None
        filename : str, optional
            The filename of the experiment, by default 'experimentdata'
        """

        if isinstance(input_data, np.ndarray) and domain is None:
            raise ValueError('If you provide a numpy array as input_data, you have to provide the domain!')

        self.filename = filename

        self.input_data = _construct_data(input_data)
        self.output_data = _construct_data(output_data)

        # Create empty output_data from indices if output_data is empty
        if self.output_data.is_empty():
            self.output_data = _Data.from_indices(self.input_data.indices)
            job_value = Status.OPEN

        else:
            job_value = Status.FINISHED

        self.domain = _construct_domain(domain, self.input_data)

        # Create empty input_data from domain if input_data is empty
        if self.input_data.is_empty():
            self.input_data = _Data.from_domain(self.domain)

        self.jobs = _construct_jobs(jobs, self.input_data, job_value)

        # Check if the columns of input_data are in the domain
        if not self.input_data.has_columnnames(self.domain.names):
            self.input_data.set_columnnames(self.domain.names)

        if self.output_data.names == [0]:
            self.output_data.set_columnnames(['y'])

    def __len__(self):
        """The len() method returns the number of datapoints"""
        return len(self.input_data)

    def __iter__(self) -> Iterator[Tuple[Dict[str, Any]]]:
        return self.input_data.__iter__()

    def __next__(self):
        return self.input_data.__next__()

    def __add__(self, other: ExperimentData | ExperimentSample) -> ExperimentData:
        """The + operator combines two ExperimentData objects"""
        # Check if the domains are the same

        if not isinstance(other, (ExperimentData, ExperimentSample)):
            raise TypeError(f"Can only add ExperimentData or ExperimentSample objects, not {type(other)}")

        if isinstance(other, ExperimentData) and self.domain != other.domain:
            raise ValueError("Cannot add ExperimentData objects with different domains")

        return ExperimentData(input_data=self.input_data + other.input_data,
                              output_data=self.output_data + other.output_data,
                              jobs=self.jobs + other.jobs, domain=self.domain,
                              filename=self.filename)

    def __eq__(self, __o: ExperimentData) -> bool:
        return all([self.input_data == __o.input_data,
                    self.output_data == __o.output_data,
                    self.jobs == __o.jobs,
                    self.domain == __o.domain])

    def __getitem__(self, index: int | slice | Iterable[int]) -> _Data:
        """The [] operator returns a single datapoint or a subset of datapoints"""
        return ExperimentData(input_data=self.input_data[index], output_data=self.output_data[index],
                              jobs=self.jobs[index], domain=self.domain, filename=self.filename)

    def _repr_html_(self) -> str:
        return self.input_data.combine_data_to_multiindex(self.output_data)._repr_html_()

    def _access_file(operation: Callable) -> Callable:
        """Wrapper for accessing a single resource with a file lock

        Returns
        -------
        decorator
        """
        @wraps(operation)
        def wrapper_func(self, *args, **kwargs) -> None:
            lock = FileLock(Path(self.filename).with_suffix('.lock'))
            with lock:
                self = ExperimentData.from_file(filename=Path(self.filename))
                value = operation(self, *args, **kwargs)
                self.store(filename=Path(self.filename))
            return value

        return wrapper_func

    #                                                      Alternative Constructors
    # =============================================================================

    @classmethod
    def from_file(cls: Type[ExperimentData], filename: str = 'experimentdata') -> ExperimentData:
        """Create an ExperimentData object from .csv and .json files.

        Parameters
        ----------
        filename : str, optional
            Name of the file, excluding suffix, by default 'experimentdata'.

        Returns
        -------
        ExperimentData
            ExperimentData object containing the loaded data.
        """
        try:
            return cls._from_file_attempt(filename)
        except FileNotFoundError:
            try:
                filename_with_path = Path(get_original_cwd()) / filename
            except ValueError:  # get_original_cwd() hydra initialization error
                raise FileNotFoundError(f"Cannot find the file {filename} !")

            return cls._from_file_attempt(filename_with_path)

    @classmethod
    def from_sampling(cls, sampler: _Sampler, filename: str = 'experimentdata') -> ExperimentData:
        """Create an ExperimentData object from a sampler.

        Parameters
        ----------
        sampler : Sampler
            Sampler object containing the sampling strategy.

        Returns
        -------
        ExperimentData
            ExperimentData object containing the sampled data.
        """
        experimentdata = sampler.get_samples()
        experimentdata.filename = filename
        return experimentdata

    @classmethod
    def from_yaml(cls, config: DictConfig) -> ExperimentData:
        """Create an ExperimentData object from a hydra yaml configuration.

        Parameters
        ----------
        config : DictConfig
            A DictConfig object containing the configuration.

        Returns
        -------
        ExperimentData
            ExperimentData object containing the loaded data.
        """
        # Option 1: From exisiting ExperimentData files
        if 'from_file' in config.experimentdata:
            return cls.from_file(filename=config.experimentdata.from_file.filepath)

        # Option 2: Sample from the domain
        elif 'from_sampling' in config.experimentdata:
            sampler = _Sampler.from_yaml(config.domain, config.experimentdata.from_sampling)
            return sampler.get_samples()
            # return cls.from_sampling(sampler)

        else:
            raise ValueError("No valid experimentdata option found in the config file!")

    @classmethod
    def _from_file_attempt(cls: Type[ExperimentData], filename: str) -> ExperimentData:
        """Attempt to create an ExperimentData object from .csv and .json files.

        Parameters
        ----------
        filename : str
            Name of the file, excluding suffix.

        Returns
        -------
        ExperimentData
            ExperimentData object containing the loaded data.

        Raises
        ------
        FileNotFoundError
            If the file cannot be found.
        """

        try:
            return cls(domain=Path(f"{filename}_domain"), input_data=Path(f"{filename}_data"),
                       output_data=Path(f"{filename}_output"), jobs=Path(f"{filename}_jobs"))
        except FileNotFoundError:
            raise FileNotFoundError(f"Cannot find the files from {filename}.")

    #                                                                        Export
    # =============================================================================

    def select(self, indices: int | slice | Iterable[int]) -> ExperimentData:
        """Select a subset of the ExperimentData object

        Parameters
        ----------
        indices : int | slice | Iterable[int]
            The indices to select.

        Returns
        -------
        ExperimentData
            The selected ExperimentData object with only the selected indices.
        """
        return self[indices]

    def store(self, filename: str = None):
        """Store the ExperimentData to disk, with checking for a lock

        Parameters
        ----------
        filename : str, optional
            filename of the files to store, without suffix
        """
        if filename is None:
            filename = self.filename

        self.input_data.store(Path(f"{filename}_data"))
        self.output_data.store(Path(f"{filename}_output"))
        self.jobs.store(Path(f"{filename}_jobs"))
        self.domain.store(Path(f"{filename}_domain"))

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert the ExperimentData object to a tuple of numpy arrays.

        Returns
        -------
        tuple
            A tuple containing two numpy arrays, the first one for input columns, and the second for output columns.
        """
        return self.input_data.to_numpy(), self.output_data.to_numpy()

    def to_xarray(self) -> xr.Dataset:
        """
        Convert the ExperimentData object to an xarray Dataset.

        Returns
        -------
        xr.Dataset
            An xarray Dataset containing the data.
        """
        return xr.Dataset({'input': self.input_data.to_xarray('input_dim'),
                           'output': self.output_data.to_xarray('output_dim')})

    def get_n_best_output(self, n_samples: int) -> ExperimentData:
        df = self.output_data.n_best_samples(n_samples, self.output_data.names)
        return self[df.index]

    #                                                         Append or remove data
    # =============================================================================

    def _add(self, domain: Optional[Domain] = None, input_data: Optional[DataTypes] = None,
             output_data: Optional[DataTypes] = None, jobs: Optional[Path | str] = None) -> None:
        self.add_experiments(ExperimentData(domain=domain, input_data=input_data, output_data=output_data, jobs=jobs))

    def add_experiments(self, experiment_sample: ExperimentSample | ExperimentData) -> None:
        """
        Add an ExperimentSample or ExperimentData to the ExperimentData attribute.

        Parameters
        ----------
        experiment_sample : ExperimentSample or ExperimentData
            Experiment(s) to add.
        """
        if isinstance(experiment_sample, ExperimentData):
            experiment_sample._reset_index()

        self.input_data += experiment_sample.input_data
        self.output_data += experiment_sample.output_data
        self.jobs += experiment_sample.jobs

    def add_input_parameter(self, name: str, parameter: Parameter) -> None:
        """Add a new input column to the ExperimentData object.

        Parameters
        ----------
        name
            name of the new input column
        parameter
            Parameter object of the new input column
        """
        self.input_data.add_column(name)
        self.domain.add(name, parameter)

    def add_output_parameter(self, name: str) -> None:
        """Add a new output column to the ExperimentData object.

        Parameters
        ----------
        name
            name of the new output column
        """
        self.output_data.add_column(name)

    def add(self, data: pd.DataFrame):
        """
        Append data to the ExperimentData object.

        Parameters
        ----------
        data : pd.DataFrame
            Data to append.
        """
        self.input_data.add(data)
        self.output_data.add_empty_rows(len(data))

        # Apparently you need to cast the types again
        # TODO: Breaks if values are NaN or infinite
        self.input_data.data = self.input_data.data.astype(
            self.domain._cast_types_dataframe())

        self.jobs.add(number_of_jobs=len(data))

    def add_numpy_arrays(self, input: np.ndarray, output: Optional[np.ndarray] = None):
        """
        Append a numpy array to the ExperimentData object.

        Parameters
        ----------
        input : np.ndarray
            2D numpy array to add to the input data.
        output : np.ndarray, optional
            2D numpy array to add to the output data. By default None.
        """
        self.input_data.add_numpy_arrays(input)

        if output is None:
            status = Status.OPEN
            self.output_data.add_empty_rows(len(input))
        else:
            status = Status.FINISHED
            self.output_data.add_numpy_arrays(output)

        self.jobs.add(number_of_jobs=len(input), status=status)

    def fill_output(self, output: np.ndarray, label: str = "y"):
        """
        Fill NaN values in the output data with the given array

        Parameters
        ----------
        output : np.ndarray
            Output data to fill
        label : str, optional
            Label of the output column to add to, by default "y".
        """
        if label not in self.output_data.names:
            self.output_data.add_column(label)

        filled_indices: Iterable[int] = self.output_data.fill_numpy_arrays(output)

        # Set the status of the filled indices to FINISHED
        self.jobs.mark_as_finished(filled_indices)

    def remove_rows_bottom(self, number_of_rows: int):
        """
        Remove a number of rows from the end of the ExperimentData object.

        Parameters
        ----------
        number_of_rows : int
            Number of rows to remove from the bottom.
        """
        if number_of_rows == 0:
            return  # Don't do anything if 0 rows need to be removed

        # get the last indices from data.data
        indices = self.input_data.data.index[-number_of_rows:]

        # remove the indices rows_to_remove from data.data
        self.input_data.remove(indices)
        self.output_data.remove(indices)
        self.jobs.remove(indices)

    def _reset_index(self) -> None:
        """
        Reset the index of the ExperimentData object.
        """
        self.input_data.reset_index()
        self.output_data.reset_index()
        self.jobs.reset_index()

    #                                                                        ExperimentSample
    # =============================================================================

    def _get_experiment_sample(self, index: int) -> ExperimentSample:
        """
        Gets the experiment_sample at the given index.

        Parameters
        ----------
        index : int
            The index of the experiment_sample to retrieve.

        Returns
        -------
        ExperimentSample
            The ExperimentSample at the given index.
        """
        return ExperimentSample(dict_input=self.input_data.get_data_dict(index),
                                dict_output=self.output_data.get_data_dict(index), jobnumber=index)

    def _set_experiment_sample(self, experiment_sample: ExperimentSample) -> None:
        """
        Sets the ExperimentSample at the given index.

        Parameters
        ----------
        experiment_sample : ExperimentSample
            The ExperimentSample to set.
        """
        for column, value in experiment_sample.output_data.items():
            self.output_data.set_data(index=experiment_sample.job_number, value=value, column=column)

        self.jobs.mark_as_finished(experiment_sample._jobnumber)

    @_access_file
    def _write_experiment_sample(self, experiment_sample: ExperimentSample) -> None:
        """
        Sets the ExperimentSample at the given index.

        Parameters
        ----------
        experiment_sample : ExperimentSample
            The ExperimentSample to set.
        """
        self._set_experiment_sample(experiment_sample)

    def _access_open_job_data(self) -> ExperimentSample:
        """Get the data of the first available open job.

        Returns
        -------
        ExperimentSample
            The ExperimentSample object of the first available open job.
        """
        job_index = self.jobs.get_open_job()
        self.jobs.mark_as_in_progress(job_index)
        experiment_sample = self._get_experiment_sample(job_index)
        return experiment_sample

    @_access_file
    def _get_open_job_data(self) -> ExperimentSample:
        """Get the data of the first available open job by
        accessing the ExperimenData on disk.

        Returns
        -------
        ExperimentSample
            The ExperimentSample object of the first available open job.
        """
        return self._access_open_job_data()

    #                                                                          Jobs
    # =============================================================================

    def _set_error(self, index: int) -> None:
        """Mark the experiment_sample at the given index as error.

        Parameters
        ----------
        index
            index of the experiment_sample to mark as error
        """
        self.jobs.mark_as_error(index)
        self.output_data.set_data(index, value='ERROR')

    @_access_file
    def _write_error(self, index: int):
        """Mark the experiment_sample at the given index as error and write to ExperimentData file.

        Parameters
        ----------
        index
            index of the experiment_sample to mark as error
        """
        self._set_error(index)

    @_access_file
    def is_all_finished(self) -> bool:
        """Check if all jobs are finished

        Returns
        -------
        bool
            True if all jobs are finished, False otherwise
        """
        return self.jobs.is_all_finished()

    def mark_all_open(self) -> None:
        """Mark all jobs as open"""
        self.jobs.mark_all_open()

    #                                                                Datageneration
    # =============================================================================

    def run(self, data_generator: _DataGenerator, mode: str = 'sequential',
            kwargs: Optional[dict] = None) -> None:
        """Run any function over the entirery of the experiments

        Parameters
        ----------
        data_generator : DataGenerator
            data grenerator to use
        mode, optional
            operational mode, by default 'sequential'
        kwargs, optional
            Any keyword arguments that need to be supplied to the function, by default None

        Raises
        ------
        ValueError
            Raised when invalid parallelization mode is specified
        """
        if kwargs is None:
            kwargs = {}

        if mode.lower() == "sequential":
            return self._run_sequential(data_generator, kwargs)
        elif mode.lower() == "parallel":
            return self._run_multiprocessing(data_generator, kwargs)
        elif mode.lower() == "cluster":
            return self._run_cluster(data_generator, kwargs)
        else:
            raise ValueError("Invalid parallelization mode specified.")

    # create an alias for the self.run function called self.evaluate
    evaluate = run

    def _run_sequential(self, data_generator: _DataGenerator, kwargs: dict):
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
            try:
                experiment_sample = self._access_open_job_data()
                logger.debug(f"Accessed experiment_sample {experiment_sample._jobnumber}")
            except NoOpenJobsError:
                logger.debug("No Open Jobs left")
                break

            try:

                # If kwargs is empty dict
                if not kwargs:
                    logger.debug(f"Running experiment_sample {experiment_sample._jobnumber}")
                else:
                    logger.debug(
                        f"Running experiment_sample {experiment_sample._jobnumber} with kwargs {kwargs}")

                _experiment_sample = data_generator.run(experiment_sample, **kwargs)  # no *args!
                self._set_experiment_sample(_experiment_sample)
            except Exception as e:
                error_msg = f"Error in experiment_sample {experiment_sample._jobnumber}: {e}"
                error_traceback = traceback.format_exc()
                logger.error(f"{error_msg}\n{error_traceback}")
                self._set_error(experiment_sample._jobnumber)

    def _run_multiprocessing(self, data_generator: _DataGenerator, kwargs: dict):
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
            try:
                experiment_sample = self._access_open_job_data()
                options.append(
                    ({'experiment_sample': experiment_sample, **kwargs},))
            except NoOpenJobsError:
                break

        def f(options: Dict[str, Any]) -> Any:
            logger.debug(f"Running experiment_sample {options['experiment_sample'].job_number}")
            return data_generator.run(**options)

        with mp.Pool() as pool:
            # maybe implement pool.starmap_async ?
            _experiment_samples: List[ExperimentSample] = pool.starmap(f, options)

        for _experiment_sample in _experiment_samples:
            self._set_experiment_sample(_experiment_sample)

    def _run_cluster(self, data_generator: _DataGenerator, kwargs: dict):
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
            self = self.from_file(self.filename)
        except FileNotFoundError:  # If not found, store current
            self.store()

        while True:
            try:
                experiment_sample = self._get_open_job_data()
            except NoOpenJobsError:
                logger.debug("No Open jobs left!")
                break

            try:
                _experiment_sample = data_generator.run(experiment_sample, **kwargs)
                self._write_experiment_sample(_experiment_sample)
            except Exception as e:
                error_msg = f"Error in experiment_sample {experiment_sample._jobnumber}: {e}"
                error_traceback = traceback.format_exc()
                logger.error(f"{error_msg}\n{error_traceback}")
                self._write_error(experiment_sample._jobnumber)
                continue

        self = self.from_file(self.filename)
        # Remove the lockfile from disk
        Path(self.filename).with_suffix('.lock').unlink(missing_ok=True)

    #                                                                  Optimization
    # =============================================================================

    def optimize(self, optimizer: _Optimizer, data_generator: _DataGenerator, iterations: int) -> None:
        if optimizer.type == 'scipy':
            self._iterate_scipy(optimizer, data_generator, iterations)
        else:
            self._iterate(optimizer, data_generator, iterations)

    def _iterate(self, optimizer: _Optimizer, data_generator: _DataGenerator,
                 iterations: int, kwargs: Optional[dict] = None):

        optimizer.set_x0(self)
        optimizer._check_number_of_datapoints()

        optimizer._construct_model(data_generator)

        for _ in range(_number_of_updates(iterations, population=optimizer.hyperparameters.population)):
            new_samples = optimizer.update_step(data_generator)
            self.add_experiments(new_samples)

            # If applicable, evaluate the new designs:
            self.run(data_generator, mode='sequential', kwargs=kwargs)

            optimizer.set_data(self)

        # Remove overiterations
        self.remove_rows_bottom(_number_of_overiterations(
            iterations, population=optimizer.hyperparameters.population))

        # Reset the optimizer
        optimizer.reset()

    def _iterate_scipy(self, optimizer: _Optimizer, data_generator: _DataGenerator,
                       iterations: int, kwargs: Optional[dict] = None):

        optimizer.set_x0(self)
        n_data_before_iterate = len(self)
        optimizer._check_number_of_datapoints()

        optimizer.run_algorithm(iterations, data_generator)

        self.add_experiments(optimizer.data)

        # TODO: At the end, the data should have n_data_before_iterate + iterations amount of elements!
        # If x_new is empty, repeat best x0 to fill up total iteration
        if len(self) == n_data_before_iterate:
            repeated_last_element = self.get_n_best_output(
                nosamples=1).to_numpy()[0].ravel()

            for repetition in range(iterations):
                self.add_experiments(ExperimentSample.from_numpy(repeated_last_element))

        # Repeat last iteration to fill up total iteration
        if len(self) < n_data_before_iterate + iterations:
            last_design = self._get_experiment_sample(len(self)-1)

            for repetition in range(iterations - (len(self) - n_data_before_iterate)):
                self.add_experiments(last_design)

        # Evaluate the function on the extra iterations
        self.run(data_generator, mode='sequential')

        # Reset the optimizer
        optimizer.reset()


def _number_of_updates(iterations: int, population: int):
    """Calculate number of update steps to acquire the correct number of iterations

    Parameters
    ----------
    iterations
        number of desired iteration steps
    population
        the population size of the optimizer

    Returns
    -------
        number of consecutive update steps
    """
    return iterations // population + (iterations % population > 0)


def _number_of_overiterations(iterations: int, population: int) -> int:
    """Calculate the number of iterations that are over the iteration limit

    Parameters
    ----------
    iterations
        number of desired iteration steos
    population
        the population size of the optimizer

    Returns
    -------
        number of iterations that are over the limit
    """
    overiterations: int = iterations % population
    if overiterations == 0:
        return overiterations
    else:
        return population - overiterations


def _construct_data(data: DataTypes) -> _Data:
    if data is None:
        return _Data()

    elif isinstance(data, _Data):
        return data

    elif isinstance(data, pd.DataFrame):
        return _Data.from_dataframe(data)

    elif isinstance(data, (Path, str)):
        return _Data.from_file(data)

    elif isinstance(data, np.ndarray):
        return _Data.from_numpy(data)

    else:
        raise TypeError(
            f"Data must be of type _Data, pd.DataFrame, np.ndarray, Path or str, not {type(data)}")


def _construct_domain(domain: Union[None, Domain], input_data: _Data) -> Domain:
    if isinstance(domain, Domain):
        return domain

    elif isinstance(domain, (Path, str)):
        return Domain.from_file(Path(domain))

    elif input_data.is_empty() and domain is None:
        return Domain()

    elif domain is None:
        return Domain.from_data(input_data)

    else:
        raise TypeError(f"Domain must be of type Domain or None, not {type(domain)}")


def _construct_jobs(jobs: Path | str | None, input_data: _Data, job_value: Status) -> _JobQueue:
    if isinstance(jobs, (Path, str)):
        return _JobQueue.from_file(Path(jobs))

    return _JobQueue.from_data(input_data, value=job_value)