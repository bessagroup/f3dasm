"""
The ExperimentData object is the main object used to store implementations of a design-of-experiments,
keep track of results, perform optimization and extract data for machine learning purposes.
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
import traceback
from functools import wraps
from pathlib import Path
from typing import (Any, Callable, Dict, Iterable, Iterator, List, Optional,
                    Tuple, Type)

# Third-party
import numpy as np
import pandas as pd
import xarray as xr
from filelock import FileLock
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from pathos.helpers import mp

# Local
from ..datageneration.datagenerator import DataGenerator
from ..datageneration.functions.function_factory import datagenerator_factory
from ..design.domain import Domain
from ..design.parameter import Parameter
from ..design.samplers import Sampler, sampler_factory
from ..logger import logger
from ..optimization import Optimizer
from ..optimization.optimizer_factory import optimizer_factory
from ._data import _Data
from ._jobqueue import NoOpenJobsError, Status, _JobQueue
from .experimentsample import ExperimentSample
from .utils import (DOMAIN_SUFFIX, INPUT_DATA_SUFFIX, JOBS_SUFFIX,
                    OUTPUT_DATA_SUFFIX, DataTypes, number_of_overiterations,
                    number_of_updates)

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class ExperimentData:
    """
    A class that contains data for experiments.
    """

    def __init__(self, domain: Optional[Domain] = None, input_data: Optional[DataTypes] = None,
                 output_data: Optional[DataTypes] = None, jobs: Optional[Path | str] = None,
                 filename: Optional[str] = 'experimentdata', path: Optional[Path] = None):
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
        path : Path, optional
            The path to the experimentdata file, by default the current working directory
        """

        if isinstance(input_data, np.ndarray) and domain is None:
            raise ValueError('If you provide a numpy array as input_data, you have to provide the domain!')

        self.filename = filename

        if path is None:
            path = Path().cwd()

        self.path = path

        self.input_data = data_factory(input_data)
        self.output_data = data_factory(output_data)

        # Create empty output_data from indices if output_data is empty
        if self.output_data.is_empty():
            self.output_data = _Data.from_indices(self.input_data.indices)
            job_value = Status.OPEN

        else:
            job_value = Status.FINISHED

        self.domain = domain_factory(domain, self.input_data)

        # Create empty input_data from domain if input_data is empty
        if self.input_data.is_empty():
            self.input_data = _Data.from_domain(self.domain)

        self.jobs = jobs_factory(jobs, self.input_data, job_value)

        # Check if the columns of input_data are in the domain
        if not self.input_data.has_columnnames(self.domain.names):
            self.input_data.set_columnnames(self.domain.names)

        # For backwards compatibility; if the output_data has only one column, rename it to 'y'
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
                              jobs=self.jobs[index], domain=self.domain, filename=self.filename, path=self.path)

    def _repr_html_(self) -> str:
        return self.input_data.combine_data_to_multiindex(self.output_data, self.jobs.to_dataframe())._repr_html_()

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
            return cls._from_file_attempt(Path(filename))
        except FileNotFoundError:
            try:
                filename_with_path = Path(get_original_cwd()) / filename
            except ValueError:  # get_original_cwd() hydra initialization error
                raise FileNotFoundError(f"Cannot find the file {filename} !")

            return cls._from_file_attempt(filename_with_path)

    @classmethod
    def from_sampling(cls, sampler: Sampler | str, domain: Domain, n_samples: int = 1,
                      seed: Optional[int] = None, filename: str = 'experimentdata') -> ExperimentData:
        """Create an ExperimentData object from a sampler.

        Parameters
        ----------
        sampler : Sampler
            Sampler object containing the sampling strategy.
        domain : Domain
            Domain object containing the domain of the experiment.
        n_samples : int, optional
            Number of samples, by default 1.
        seed : int, optional
            Seed for the random number generator, by default None.
        filename : str, optional
            Name of the file, excluding suffix, by default 'experimentdata'.

        Returns
        -------
        ExperimentData
            ExperimentData object containing the sampled data.
        """
        experimentdata = cls(domain=domain, filename=filename)
        experimentdata.sample(sampler=sampler, n_samples=n_samples, seed=seed)
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
            return cls.from_file(filename=config.experimentdata.from_file)

        # Option 2: Sample from the domain
        elif 'from_sampling' in config.experimentdata:
            domain = Domain.from_yaml(config.domain)
            return cls.from_sampling(sampler=config.experimentdata.from_sampling.sampler, domain=domain,
                                     n_samples=config.experimentdata.from_sampling.n_samples,
                                     seed=config.experimentdata.from_sampling.seed,
                                     filename=config.experimentdata.name)

        else:
            return cls(**config)

    @classmethod
    def _from_file_attempt(cls: Type[ExperimentData], filename: Path) -> ExperimentData:
        """Attempt to create an ExperimentData object from .csv and .json files.

        Parameters
        ----------
        filename : Path
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
            return cls(domain=Path(f"{filename}{DOMAIN_SUFFIX}"), input_data=Path(f"{filename}{INPUT_DATA_SUFFIX}"),
                       output_data=Path(f"{filename}{OUTPUT_DATA_SUFFIX}"), jobs=Path(f"{filename}{JOBS_SUFFIX}"),
                       filename=filename.name, path=filename.parent)
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

        self.input_data.store(Path(f"{filename}{INPUT_DATA_SUFFIX}"))
        self.output_data.store(Path(f"{filename}{OUTPUT_DATA_SUFFIX}"))
        self.jobs.store(Path(f"{filename}{JOBS_SUFFIX}"))
        self.domain.store(Path(f"{filename}{DOMAIN_SUFFIX}"))

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert the ExperimentData object to a tuple of numpy arrays.

        Returns
        -------
        tuple
            A tuple containing two numpy arrays, the first one for input columns, and the second for output columns.
        """
        return self.input_data.to_numpy(), self.output_data.to_numpy()

    def to_pandas(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Convert the ExperimentData object to a pandas DataFrame.

        Returns
        -------
        tuple
            A tuple containing two pandas DataFrames, the first one for input columns, and the second for output
        """
        return self.input_data.to_dataframe(), self.output_data.to_dataframe()

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
        """Get the n best samples from the output data. We consider a minimization problem

        Parameters
        ----------
        n_samples : int
            Number of samples to select.

        Returns
        -------
        ExperimentData
            New experimentData object with a selection of the n best samples.
        """
        df = self.output_data.n_best_samples(n_samples, self.output_data.names)
        return self[df.index]

    #                                                         Append or remove data
    # =============================================================================

    def add(self, domain: Optional[Domain] = None, input_data: Optional[DataTypes] = None,
            output_data: Optional[DataTypes] = None, jobs: Optional[Path | str] = None) -> None:
        """Add data to the ExperimentData object.

        Parameters
        ----------
        domain : Optional[Domain], optional
            Domain of the added object, by default None
        input_data : Optional[DataTypes], optional
            input parameters of the added object, by default None
        output_data : Optional[DataTypes], optional
            output parameters of the added object, by default None
        jobs : Optional[Path  |  str], optional
            jobs off the added object, by default None
        """
        self._add_experiments(ExperimentData(domain=domain, input_data=input_data, output_data=output_data, jobs=jobs))

    def _add_experiments(self, experiment_sample: ExperimentSample | ExperimentData) -> None:
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

        # Check if indices of the internal objects are equal
        if not (self.input_data.indices.equals(self.output_data.indices)
                and self.input_data.indices.equals(self.jobs.indices)):
            raise ValueError(f"Indices of the internal objects are not equal."
                             f"input_data {self.input_data.indices}, output_data {self.output_data.indices},"
                             f"jobs: {self.jobs.indices}")

        # Apparently you need to cast the types again
        # TODO: Breaks if values are NaN or infinite
        self.input_data.data = self.input_data.data.astype(
            self.domain._cast_types_dataframe())

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
            self.add_output_parameter(label)

        filled_indices: Iterable[int] = self.output_data.fill_numpy_arrays(output)

        # Set the status of the filled indices to FINISHED
        self.jobs.mark(filled_indices, Status.FINISHED)

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
                                dict_output=self.output_data.get_data_dict(index), jobnumber=index,
                                experimentdata_directory=self.path)

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

        self.jobs.mark(experiment_sample._jobnumber, status=Status.FINISHED)

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
        self.jobs.mark(job_index, status=Status.IN_PROGRESS)
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
        # self.jobs.mark_as_error(index)
        self.jobs.mark(index, status=Status.ERROR)
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

    def mark(self, indices: Iterable[int], status: str) -> None:
        """Mark the jobs at the given indices with the given status.

        Parameters
        ----------
        indices : Iterable[int]
            indices of the jobs to mark
        status : str
            status to mark the jobs with: choose between: 'open', 'in progress', 'finished' or 'error'

        Raises
        ------
        ValueError
            If the given status is not any of 'open', 'in progress', 'finished' or 'error'
        """
        # Check if the status is in Status
        if not any(status.lower() == s.value for s in Status):
            raise ValueError(f"Invalid status {status} given. "
                             f"\nChoose from values: {', '.join([s.value for s in Status])}")

        self.jobs.mark(indices, status)

    def mark_all(self, status: str) -> None:
        """Mark all the experiments with the given status

        Parameters
        ----------
        status : str
            status to mark the jobs with: choose between: 'open', 'in progress', 'finished' or 'error'

        Raises
        ------
        ValueError
            If the given status is not any of 'open', 'in progress', 'finished' or 'error'
        """
        self.mark(self.jobs.indices, status)

    def mark_all_error_open(self) -> None:
        """
        Mark all the experiments that have the status 'error' open
        """
        self.jobs.mark_all_error_open()
    #                                                                Datageneration
    # =============================================================================

    def evaluate(self, data_generator: DataGenerator, mode: str = 'sequential',
                 kwargs: Optional[dict] = None) -> None:
        """Run any function over the entirety of the experiments

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

        if isinstance(data_generator, str):
            data_generator = datagenerator_factory(data_generator, self.domain, kwargs)

        if mode.lower() == "sequential":
            return self._run_sequential(data_generator, kwargs)
        elif mode.lower() == "parallel":
            return self._run_multiprocessing(data_generator, kwargs)
        elif mode.lower() == "cluster":
            return self._run_cluster(data_generator, kwargs)
        else:
            raise ValueError("Invalid parallelization mode specified.")

    def _run_sequential(self, data_generator: DataGenerator, kwargs: dict):
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

    def _run_multiprocessing(self, data_generator: DataGenerator, kwargs: dict):
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

    def _run_cluster(self, data_generator: DataGenerator, kwargs: dict):
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

    def optimize(self, optimizer: Optimizer | str, data_generator: DataGenerator | str,
                 iterations: int, kwargs: Optional[Dict[str, Any]] = None,
                 hyperparameters: Optional[Dict[str, Any]] = None) -> None:
        """Optimize the experimentdata object

        Parameters
        ----------
        optimizer : Optimizer
            Optimizer object to use
        data_generator : DataGenerator | str
            Data generator object to use
        iterations : int
            Number of iterations to run
        kwargs : Dict[str, Any], optional
            Any additional keyword arguments that need to be supplied to the data generator, by default None
        hyperparameters : Dict[str, Any], optional
            Any additional hyperparameters that need to be supplied to the optimizer, by default None

        Raises
        ------
        ValueError
            Raised when invalid optimizer type is specified
        """
        if isinstance(data_generator, str):
            data_generator: DataGenerator = datagenerator_factory(data_generator, self.domain, kwargs)

        if isinstance(optimizer, str):
            optimizer: Optimizer = optimizer_factory(optimizer, self.domain, hyperparameters)

        if optimizer.type == 'scipy':
            self._iterate_scipy(optimizer, data_generator, iterations, kwargs)
        else:
            self._iterate(optimizer, data_generator, iterations, kwargs)

    def _iterate(self, optimizer: Optimizer, data_generator: DataGenerator,
                 iterations: int, kwargs: Optional[dict] = None):
        """Internal represenation of the iteration process

        Parameters
        ----------
        optimizer : Optimizer
            Optimizer object
        data_generator : DataGenerator
            DataGenerator object
        iterations : int
            number of iterations
        kwargs : Optional[dict], optional
            any additional keyword arguments that will be passed to the DataGenerator, by default None
        """
        optimizer.set_x0(self)
        optimizer._check_number_of_datapoints()

        optimizer._construct_model(data_generator)

        for _ in range(number_of_updates(iterations, population=optimizer.hyperparameters.population)):
            new_samples = optimizer.update_step(data_generator)

            # If new_samples is a tuple of input_data and output_data
            if isinstance(new_samples, tuple):
                self.add(domain=self.domain, input_data=new_samples[0], output_data=new_samples[1])

            else:
                self._add_experiments(new_samples)

            # If applicable, evaluate the new designs:
            self.evaluate(data_generator, mode='sequential', kwargs=kwargs)

            optimizer.set_data(self)

        # Remove overiterations
        self.remove_rows_bottom(number_of_overiterations(
            iterations, population=optimizer.hyperparameters.population))

        # Reset the optimizer
        optimizer.reset(ExperimentData(domain=self.domain))

    def _iterate_scipy(self, optimizer: Optimizer, data_generator: DataGenerator,
                       iterations: int, kwargs: Optional[dict] = None):
        """Internal represenation of the iteration process for scipy-optimize algorithms

        Parameters
        ----------
        optimizer : _Optimizer
            Optimizer object
        data_generator : DataGenerator
            DataGenerator object
        iterations : int
            number of iterations
        kwargs : Optional[dict], optional
            any additional keyword arguments that will be passed to the DataGenerator, by default None
        """

        optimizer.set_x0(self)
        n_data_before_iterate = len(self)
        optimizer._check_number_of_datapoints()

        optimizer.run_algorithm(iterations, data_generator)

        self._add_experiments(optimizer.data)

        # TODO: At the end, the data should have n_data_before_iterate + iterations amount of elements!
        # If x_new is empty, repeat best x0 to fill up total iteration
        if len(self) == n_data_before_iterate:
            repeated_last_element = self.get_n_best_output(
                nosamples=1).to_numpy()[0].ravel()

            for repetition in range(iterations):
                self._add_experiments(ExperimentSample.from_numpy(repeated_last_element))

        # Repeat last iteration to fill up total iteration
        if len(self) < n_data_before_iterate + iterations:
            last_design = self._get_experiment_sample(len(self)-1)

            for repetition in range(iterations - (len(self) - n_data_before_iterate)):
                self._add_experiments(last_design)

        # Evaluate the function on the extra iterations
        self.evaluate(data_generator, mode='sequential')

        # Reset the optimizer
        optimizer.reset(ExperimentData(domain=self.domain))

    #                                                                      Sampling
    # =============================================================================

    def sample(self, sampler: Sampler | str, n_samples: int = 1, seed: Optional[int] = None) -> None:
        """Sample data from the domain providing the sampler strategy

        Parameters
        ----------
        sampler : Sampler or str
            Sampler callable or string of built-in sampler
        n_samples : int, optional
            Number of samples to generate, by default 1
        seed : Optional[int], optional
            Seed to use for the sampler, by default None

        Note
        ----
        If a string is passed, it should be one of the built-in samplers:
        - 'random' : Random sampling
        - 'latin' : Latin Hypercube Sampling
        - 'sobol' : Sobol Sequence Sampling

        Raises
        ------
        ValueError
            Raised when invalid sampler type is specified
        """

        if isinstance(sampler, str):
            sampler = sampler_factory(sampler, self.domain)

        sample_data: DataTypes = sampler(domain=self.domain, n_samples=n_samples, seed=seed)
        self.add(input_data=sample_data, domain=self.domain)


def data_factory(data: DataTypes) -> _Data:
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


def domain_factory(domain: Domain | None, input_data: _Data) -> Domain:
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


def jobs_factory(jobs: Path | str | _JobQueue | None, input_data: _Data, job_value: Status) -> _JobQueue:
    """Creates a _JobQueue object from particular inpute

    Parameters
    ----------
    jobs : Path | str | None
        input data for the jobs
    input_data : _Data
        _Data object to extract indices from, if necessary
    job_value : Status
        initial value of all the jobs

    Returns
    -------
    _JobQueue
        JobQueue object
    """
    if isinstance(jobs, _JobQueue):
        return jobs

    if isinstance(jobs, (Path, str)):
        return _JobQueue.from_file(Path(jobs))

    return _JobQueue.from_data(input_data, value=job_value)
