"""
The ExperimentData object is the main object used to store implementations
 of a design-of-experiments, keep track of results, perform optimization and
 extract data for machine learning purposes.
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
import inspect
import traceback
from copy import copy
from functools import wraps
from pathlib import Path
from time import sleep
from typing import (Any, Callable, Dict, Iterable, Iterator, List, Literal,
                    Optional, Tuple, Type)

# Third-party
import numpy as np
import pandas as pd
import xarray as xr
from filelock import FileLock
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from pathos.helpers import mp

# Local
from ..datageneration.datagenerator import DataGenerator, convert_function
from ..datageneration.functions.function_factory import _datagenerator_factory
from ..design.domain import Domain, _domain_factory
from ..logger import logger
from ..optimization import Optimizer
from ..optimization.optimizer_factory import _optimizer_factory
from ._data import DataTypes, _Data, _data_factory
from ._io import (DOMAIN_FILENAME, EXPERIMENTDATA_SUBFOLDER,
                  INPUT_DATA_FILENAME, JOBS_FILENAME, LOCK_FILENAME, MAX_TRIES,
                  OUTPUT_DATA_FILENAME, _project_dir_factory)
from ._jobqueue import NoOpenJobsError, Status, _jobs_factory
from .experimentsample import ExperimentSample
from .samplers import Sampler, SamplerNames, _sampler_factory
from .utils import number_of_overiterations, number_of_updates

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

    def __init__(self,
                 domain: Optional[Domain] = None,
                 input_data: Optional[DataTypes] = None,
                 output_data: Optional[DataTypes] = None,
                 jobs: Optional[Path | str] = None,
                 project_dir: Optional[Path] = None):
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
        project_dir : Path | str, optional
            A user-defined directory where the f3dasm project folder will be \
            created, by default the current working directory.

        Note
        ----

        The following data formats are supported for input and output data:

        * numpy array
        * pandas Dataframe
        * path to a csv file

        If no domain object is provided, the domain is inferred from the \
        input_data.

        If the provided project_dir does not exist, it will be created.

        Raises
        ------

        ValueError
            If the input_data is a numpy array, the domain has to be provided.
        """

        if isinstance(input_data, np.ndarray) and domain is None:
            raise ValueError(
                'If you provide a numpy array as input_data, \
                you have to provide the domain!')

        self.project_dir = _project_dir_factory(project_dir)

        self._input_data = _data_factory(input_data)
        self._output_data = _data_factory(output_data)

        # Create empty output_data from indices if output_data is empty
        if self._output_data.is_empty():
            self._output_data = _Data.from_indices(self._input_data.indices)
            job_value = Status.OPEN

        else:
            job_value = Status.FINISHED

        self.domain = _domain_factory(
            domain=domain, input_data=self._input_data.to_dataframe(),
            output_data=self._output_data.to_dataframe())

        # Create empty input_data from domain if input_data is empty
        if self._input_data.is_empty():
            self._input_data = _Data.from_domain(self.domain)

        self._jobs = _jobs_factory(
            jobs, self._input_data, self._output_data, job_value)

        # Check if the columns of input_data are in the domain
        if not self._input_data.has_columnnames(self.domain.names):
            self._input_data.set_columnnames(self.domain.names)

        if not self._output_data.has_columnnames(self.domain.output_names):
            self._output_data.set_columnnames(self.domain.output_names)

        # For backwards compatibility; if the output_data has
        #  only one column, rename it to 'y'
        if self._output_data.names == [0]:
            self._output_data.set_columnnames(['y'])

    def __len__(self):
        """The len() method returns the number of datapoints"""
        if self._input_data.is_empty():
            return len(self._output_data)

        return len(self._input_data)

    def __iter__(self) -> Iterator[Tuple[Dict[str, Any]]]:
        self.current_index = 0
        return self

    def __next__(self) -> ExperimentSample:
        if self.current_index >= len(self):
            raise StopIteration
        else:
            index = self.index[self.current_index]
            self.current_index += 1
            return self.get_experiment_sample(index)

    def __add__(self,
                __o: ExperimentData | ExperimentSample) -> ExperimentData:
        """The + operator combines two ExperimentData objects"""
        # Check if the domains are the same

        if not isinstance(__o, (ExperimentData, ExperimentSample)):
            raise TypeError(
                f"Can only add ExperimentData or "
                f"ExperimentSample objects, not {type(__o)}")

        return ExperimentData(
            input_data=self._input_data + __o._input_data,
            output_data=self._output_data + __o._output_data,
            jobs=self._jobs + __o._jobs, domain=self.domain + __o.domain,
            project_dir=self.project_dir)

    def __eq__(self, __o: ExperimentData) -> bool:
        return all([self._input_data == __o._input_data,
                    self._output_data == __o._output_data,
                    self._jobs == __o._jobs,
                    self.domain == __o.domain])

    def _repr_html_(self) -> str:
        return self._input_data.combine_data_to_multiindex(
            self._output_data, self._jobs.to_dataframe())._repr_html_()

    def __repr__(self) -> str:
        return self._input_data.combine_data_to_multiindex(
            self._output_data, self._jobs.to_dataframe()).__repr__()

    def _access_file(operation: Callable) -> Callable:
        """Wrapper for accessing a single resource with a file lock

        Parameters
        ----------
        operation : Callable
            The operation to be performed on the resource

        Returns
        -------
        Callable
            The wrapped operation
        """
        @wraps(operation)
        def wrapper_func(self: ExperimentData, *args, **kwargs) -> None:
            lock = FileLock(
                (self.
                 project_dir / EXPERIMENTDATA_SUBFOLDER / LOCK_FILENAME)
                .with_suffix('.lock'))

            # If the lock has been acquired:
            with lock:
                tries = 0
                while tries < MAX_TRIES:
                    try:
                        self = ExperimentData.from_file(self.project_dir)
                        value = operation(self, *args, **kwargs)
                        self.store()
                        break

                    # Racing conditions can occur when the file is empty
                    # and the file is being read at the same time
                    except pd.errors.EmptyDataError:
                        tries += 1
                        logger.debug((
                            f"EmptyDataError occurred, retrying"
                            f" {tries+1}/{MAX_TRIES}"))
                        sleep(1)

                    raise pd.errors.EmptyDataError()

            return value

        return wrapper_func
    #                                                                Properties
    # =========================================================================

    @property
    def index(self) -> pd.Index:
        """Returns an iterable of the job number of the experiments

        Returns
        -------
        pd.Index
            The job number of all the experiments in pandas Index format
        """
        if self._input_data.is_empty():
            return self._output_data.indices

        return self._input_data.indices

    #                                                  Alternative Constructors
    # =========================================================================

    @classmethod
    def from_file(cls: Type[ExperimentData],
                  project_dir: Path | str) -> ExperimentData:
        """Create an ExperimentData object from .csv and .json files.

        Parameters
        ----------
        project_dir : Path | str
            User defined path of the experimentdata directory.

        Returns
        -------
        ExperimentData
            ExperimentData object containing the loaded data.
        """
        if isinstance(project_dir, str):
            project_dir = Path(project_dir)

        try:
            return cls._from_file_attempt(project_dir)
        except FileNotFoundError:
            try:
                filename_with_path = Path(get_original_cwd()) / project_dir
            except ValueError:  # get_original_cwd() hydra initialization error
                raise FileNotFoundError(
                    f"Cannot find the folder {project_dir} !")

            return cls._from_file_attempt(filename_with_path)

    @classmethod
    def from_sampling(cls, sampler: Sampler | str, domain: Domain | DictConfig,
                      n_samples: int = 1,
                      seed: Optional[int] = None,
                      **kwargs) -> ExperimentData:
        """Create an ExperimentData object from a sampler.

        Parameters
        ----------
        sampler : Sampler | str
            Sampler object containing the sampling strategy or one of the
            built-in sampler names.
        domain : Domain | DictConfig
            Domain object containing the domain of the experiment or hydra
            DictConfig object containing the configuration.
        n_samples : int, optional
            Number of samples, by default 1.
        seed : int, optional
            Seed for the random number generator, by default None.

        Returns
        -------
        ExperimentData
            ExperimentData object containing the sampled data.

        Note
        ----

        If a string is passed for the sampler argument, it should be one
        of the built-in samplers:

        * 'random' : Random sampling
        * 'latin' : Latin Hypercube Sampling
        * 'sobol' : Sobol Sequence Sampling
        * 'grid' : Grid Search Sampling

        Any additional keyword arguments are passed to the sampler.
        """
        experimentdata = cls(domain=domain)
        experimentdata.sample(
            sampler=sampler, n_samples=n_samples, seed=seed, **kwargs)
        return experimentdata

    @classmethod
    def from_yaml(cls, config: DictConfig) -> ExperimentData:
        """Create an ExperimentData object from a hydra yaml configuration.

        Parameters
        ----------
        config : DictConfig
            A DictConfig object containing the configuration of the \
            experiment data.

        Returns
        -------
        ExperimentData
            ExperimentData object containing the loaded data.
        """
        # Option 0: Both existing and sampling
        if 'from_file' in config and 'from_sampling' in config:
            return cls.from_file(config.from_file) + cls.from_sampling(
                **config.from_sampling)

        # Option 1: From exisiting ExperimentData files
        if 'from_file' in config:
            return cls.from_file(config.from_file)

        # Option 2: Sample from the domain
        if 'from_sampling' in config:
            return cls.from_sampling(**config.from_sampling)

        else:
            return cls(**config)

    @classmethod
    def _from_file_attempt(cls: Type[ExperimentData],
                           project_dir: Path) -> ExperimentData:
        """Attempt to create an ExperimentData object
        from .csv and .pkl files.

        Parameters
        ----------
        path : Path
            Name of the user-defined directory where the files are stored.

        Returns
        -------
        ExperimentData
            ExperimentData object containing the loaded data.

        Raises
        ------
        FileNotFoundError
            If the files cannot be found.
        """
        subdirectory = project_dir / EXPERIMENTDATA_SUBFOLDER

        try:
            return cls(domain=subdirectory / DOMAIN_FILENAME,
                       input_data=subdirectory / INPUT_DATA_FILENAME,
                       output_data=subdirectory / OUTPUT_DATA_FILENAME,
                       jobs=subdirectory / JOBS_FILENAME,
                       project_dir=project_dir)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Cannot find the files from {subdirectory}.")

    #                                                         Selecting subsets
    # =========================================================================

    def select(self, indices: int | Iterable[int]) -> ExperimentData:
        """Select a subset of the ExperimentData object

        Parameters
        ----------
        indices : int | Iterable[int]
            The indices to select.

        Returns
        -------
        ExperimentData
            The selected ExperimentData object with only the selected indices.
        """

        return ExperimentData(input_data=self._input_data[indices],
                              output_data=self._output_data[indices],
                              jobs=self._jobs[indices],
                              domain=self.domain, project_dir=self.project_dir)

    def drop_output(self, names: Iterable[str] | str) -> ExperimentData:
        """Drop a column from the output data

        Parameters
        ----------
        names : Iteraeble | str
            The names of the columns to drop.

        Returns
        -------
        ExperimentData
            The ExperimentData object with the column dropped.
        """
        return ExperimentData(input_data=self._input_data,
                              output_data=self._output_data.drop(names),
                              jobs=self._jobs, domain=self.domain.drop_output(
                                  names),
                              project_dir=self.project_dir)

    def select_with_status(self, status: Literal['open', 'in progress',
                                                 'finished', 'error']
                           ) -> ExperimentData:
        """Select a subset of the ExperimentData object with a given status

        Parameters
        ----------
        status : Literal['open', 'in progress', 'finished', 'error']
            The status to select.

        Returns
        -------
        ExperimentData
            The selected ExperimentData object with only the selected status.

        Raises
        ------
        ValueError
            Raised when invalid status is specified
        """
        if status not in [s.value for s in Status]:
            raise ValueError(f"Invalid status {status} given. "
                             f"\nChoose from values: "
                             f"{', '.join([s.value for s in Status])}")

        _indices = self._jobs.select_all(status).indices
        return self.select(_indices)

    def get_input_data(self,
                       parameter_names: Optional[str | Iterable[str]] = None
                       ) -> ExperimentData:
        """Retrieve a subset of the input data from the ExperimentData object

        Parameters
        ----------
        parameter_names : str | Iterable[str], optional
            The name(s) of the input parameters that you want to retrieve, \
            if None all input parameters are retrieved, by default None

        Returns
        -------
        ExperimentData
            The selected ExperimentData object with only the\
             selected input data.

        Note
        ----
        If parameter_names is None, all input data is retrieved. \
        The returned ExperimentData object has the domain of \
        the original ExperimentData object, \
        but only with the selected input parameters.\
        """
        if parameter_names is None:
            return ExperimentData(input_data=self._input_data,
                                  jobs=self._jobs,
                                  domain=self.domain.select(self.domain.names),
                                  project_dir=self.project_dir)
        else:
            return ExperimentData(input_data=self._input_data.select_columns(
                parameter_names),
                jobs=self._jobs,
                domain=self.domain.select(parameter_names),
                project_dir=self.project_dir)

    def get_output_data(self,
                        parameter_names: Optional[str | Iterable[str]] = None
                        ) -> ExperimentData:
        """Retrieve a subset of the output data from the ExperimentData object

        Parameters
        ----------
        parameter_names : str | Iterable[str], optional
            The name(s) of the output parameters that you want to retrieve, \
            if None all output parameters are retrieved, by default None

        Returns
        -------
        ExperimentData
            The selected ExperimentData object with only \
            the selected output data.

        Note
        ----
        If parameter_names is None, all output data is retrieved. \
        The returned ExperimentData object has no domain object and \
        no input data!
        """
        if parameter_names is None:
            # TODO: Make a domain where space is empty
            # but it tracks output_space!
            return ExperimentData(
                output_data=self._output_data, jobs=self._jobs,
                project_dir=self.project_dir)
        else:
            return ExperimentData(
                output_data=self._output_data.select_columns(parameter_names),
                jobs=self._jobs,
                project_dir=self.project_dir)

    #                                                                    Export
    # =========================================================================

    def store(self, project_dir: Optional[Path | str] = None):
        """Write the ExperimentData to disk in the project directory.

        Parameters
        ----------
        project_dir : Optional[Path | str], optional
            The f3dasm project directory to store the \
            ExperimentData object to, by default None.

        Note
        ----
        If no project directory is provided, the ExperimentData object is \
        stored in the directory provided by the `.project_dir` attribute that \
        is set upon creation of the object.

        The ExperimentData object is stored in a subfolder 'experiment_data'.

        The ExperimentData object is stored in four files:

        * the input data (`input.csv`)
        * the output data (`output.csv`)
        * the jobs (`jobs.pkl`)
        * the domain (`domain.pkl`)

        To avoid the ExperimentData to be written simultaneously by multiple \
        processes, a '.lock' file is automatically created \
        in the project directory. Concurrent process can only sequentially \
        access the lock file. This lock file is removed after the \
        ExperimentData object is written to disk.
        """
        if project_dir is not None:
            self.set_project_dir(project_dir)

        subdirectory = self.project_dir / EXPERIMENTDATA_SUBFOLDER

        # Create the subdirectory if it does not exist
        subdirectory.mkdir(parents=True, exist_ok=True)

        self._input_data.store(subdirectory / Path(INPUT_DATA_FILENAME))
        self._output_data.store(subdirectory / Path(OUTPUT_DATA_FILENAME))
        self._jobs.store(subdirectory / Path(JOBS_FILENAME))
        self.domain.store(subdirectory / Path(DOMAIN_FILENAME))

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert the ExperimentData object to a tuple of numpy arrays.

        Returns
        -------
        tuple
            A tuple containing two numpy arrays, \
            the first one for input columns, \
            and the second for output columns.
        """
        return self._input_data.to_numpy(), self._output_data.to_numpy()

    def to_pandas(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Convert the ExperimentData object to a pandas DataFrame.

        Returns
        -------
        tuple
            A tuple containing two pandas DataFrames, \
            the first one for input columns, and the second for output
        """
        return (self._input_data.to_dataframe(),
                self._output_data.to_dataframe())

    def to_xarray(self) -> xr.Dataset:
        """
        Convert the ExperimentData object to an xarray Dataset.

        Returns
        -------
        xarray.Dataset
            An xarray Dataset containing the data.
        """
        return xr.Dataset(
            {'input': self._input_data.to_xarray('input_dim'),
             'output': self._output_data.to_xarray('output_dim')})

    def get_n_best_output(self, n_samples: int) -> ExperimentData:
        """Get the n best samples from the output data. \
        We consider lower values to be better.

        Parameters
        ----------
        n_samples : int
            Number of samples to select.

        Returns
        -------
        ExperimentData
            New experimentData object with a selection of the n best samples.

        Note
        ----

        The n best samples are selected based on the output data. \
        The output data is sorted based on the first output parameter. \
        The n best samples are selected based on this sorting. \
        """
        df = self._output_data.n_best_samples(
            n_samples, self._output_data.names)
        return self.select(df.index)

    #                                                     Append or remove data
    # =========================================================================

    def add(self, domain: Optional[Domain] = None,
            input_data: Optional[DataTypes] = None,
            output_data: Optional[DataTypes] = None,
            jobs: Optional[Path | str] = None) -> None:
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
        self.add_experiments(ExperimentData(
            domain=domain, input_data=input_data,
            output_data=output_data,
            jobs=jobs))

    def add_experiments(self,
                        experiment_sample: ExperimentSample | ExperimentData
                        ) -> None:
        """
        Add an ExperimentSample or ExperimentData to the ExperimentData
        attribute.

        Parameters
        ----------
        experiment_sample : ExperimentSample or ExperimentData
            Experiment(s) to add.

        Raises
        ------
        ValueError
            If -after checked- the indices of the input and output data
            objects are not equal.
        """

        if isinstance(experiment_sample, ExperimentData):
            experiment_sample._reset_index()
            self.domain += experiment_sample.domain

        self._input_data += experiment_sample._input_data
        self._output_data += experiment_sample._output_data
        self._jobs += experiment_sample._jobs

        # Check if indices of the internal objects are equal
        if not (self._input_data.indices.equals(self._output_data.indices)
                and self._input_data.indices.equals(self._jobs.indices)):
            raise ValueError(f"Indices of the internal objects are not equal."
                             f"input_data {self._input_data.indices}, "
                             f"output_data {self._output_data.indices},"
                             f"jobs: {self._jobs.indices}")

        # Apparently you need to cast the types again
        # TODO: Breaks if values are NaN or infinite
        # self._input_data.cast_types(self.domain)

    def overwrite(
        self, indices: Iterable[int],
            domain: Optional[Domain] = None,
            input_data: Optional[DataTypes] = None,
            output_data: Optional[DataTypes] = None,
            jobs: Optional[Path | str] = None,
            add_if_not_exist: bool = False
    ) -> None:
        """Overwrite the ExperimentData object.

        Parameters
        ----------
        indices : Iterable[int]
            The indices to overwrite.
        domain : Optional[Domain], optional
            Domain of the new object, by default None
        input_data : Optional[DataTypes], optional
            input parameters of the new object, by default None
        output_data : Optional[DataTypes], optional
            output parameters of the new object, by default None
        jobs : Optional[Path  |  str], optional
            jobs off the new object, by default None
        add_if_not_exist : bool, optional
            If True, the new objects are added if the requested indices
            do not exist in the current ExperimentData object, by default False
        """

        # Be careful, if a job has output data and gets overwritten with a
        # job that has no output data, the status is set to open. But the job
        # will still have the output data!

        # This is usually not a problem, because the output data will be
        # immediately overwritten in optimization.

        self._overwrite_experiments(
            indices=indices,
            experiment_sample=ExperimentData(
                domain=domain, input_data=input_data,
                output_data=output_data,
                jobs=jobs),
            add_if_not_exist=add_if_not_exist)

    def _overwrite_experiments(
        self, indices: Iterable[int],
            experiment_sample: ExperimentSample | ExperimentData,
            add_if_not_exist: bool) -> None:
        """
        Overwrite the ExperimentData object at the given indices.

        Parameters
        ----------
        indices : Iterable[int]
            The indices to overwrite.
        experimentdata : ExperimentData | ExperimentSample
            The new ExperimentData object to overwrite with.
        add_if_not_exist : bool
            If True, the new objects are added if the requested indices
            do not exist in the current ExperimentData object.
        """
        if not all(pd.Index(indices).isin(self.index)):
            if add_if_not_exist:
                self.add_experiments(experiment_sample)
                return
            else:
                raise ValueError(
                    f"The given indices {indices} do not exist in the current "
                    f"ExperimentData object. "
                    f"If you want to add the new experiments, "
                    f"set add_if_not_exist to True.")

        self._input_data.overwrite(
            indices=indices, other=experiment_sample._input_data)
        self._output_data.overwrite(
            indices=indices, other=experiment_sample._output_data)

        self._jobs.overwrite(
            indices=indices, other=experiment_sample._jobs)

        if isinstance(experiment_sample, ExperimentData):
            self.domain += experiment_sample.domain

    @_access_file
    def overwrite_disk(
        self, indices: Iterable[int],
            domain: Optional[Domain] = None,
            input_data: Optional[DataTypes] = None,
            output_data: Optional[DataTypes] = None,
            jobs: Optional[Path | str] = None,
            add_if_not_exist: bool = False
    ) -> None:
        self.overwrite(indices=indices, domain=domain, input_data=input_data,
                       output_data=output_data, jobs=jobs,
                       add_if_not_exist=add_if_not_exist)

    def add_input_parameter(
        self, name: str,
        type: Literal['float', 'int', 'category', 'constant'],
            **kwargs):
        """Add a new input column to the ExperimentData object.

        Parameters
        ----------
        name
            name of the new input column
        type
            type of the new input column: float, int, category or constant
        kwargs
            additional arguments for the new parameter
        """
        self._input_data.add_column(name)
        self.domain.add(name=name, type=type, **kwargs)

    def add_output_parameter(
            self, name: str, is_disk: bool, exist_ok: bool = False) -> None:
        """Add a new output column to the ExperimentData object.

        Parameters
        ----------
        name
            name of the new output column
        is_disk
            Whether the output column will be stored on disk or not
        exist_ok
            If True, it will not raise an error if the output column already
            exists, by default False
        """
        self._output_data.add_column(name, exist_ok=exist_ok)
        self.domain.add_output(name=name, to_disk=is_disk, exist_ok=exist_ok)

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
        indices = self.index[-number_of_rows:]

        # remove the indices rows_to_remove from data.data
        self._input_data.remove(indices)
        self._output_data.remove(indices)
        self._jobs.remove(indices)

    def _reset_index(self) -> None:
        """
        Reset the index of the ExperimentData object.
        """
        self._input_data.reset_index()

        if self._input_data.is_empty():
            self._output_data.reset_index()
        else:
            self._output_data.reset_index(self._input_data.indices)
        self._jobs.reset_index()

    def join(self, other: ExperimentData) -> ExperimentData:
        """Join two ExperimentData objects.

        Parameters
        ----------
        other : ExperimentData
            The other ExperimentData object to join with.

        Returns
        -------
        ExperimentData
            The joined ExperimentData object.
        """
        return ExperimentData(
            input_data=self._input_data.join(other._input_data),
            output_data=self._output_data.join(other._output_data),
            jobs=self._jobs,
            domain=self.domain + other.domain,
            project_dir=self.project_dir)
#                                                                  ExperimentSample
    # =============================================================================

    def get_experiment_sample(self, index: int) -> ExperimentSample:
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
        output_experiment_sample_dict = self._output_data.get_data_dict(index)

        dict_output = {k: (v, self.domain.output_space[k].to_disk)
                       for k, v in output_experiment_sample_dict.items()}

        return ExperimentSample(dict_input=self._input_data.get_data_dict(
            index),
            dict_output=dict_output,
            jobnumber=index,
            experimentdata_directory=self.project_dir)

    def get_experiment_samples(
            self,
            indices: Optional[Iterable[int]] = None) -> List[ExperimentSample]:
        """
        Gets the experiment_samples at the given indices.

        Parameters
        ----------
        indices : Optional[Iterable[int]], optional
            The indices of the experiment_samples to retrieve, by default None
            If None, all experiment_samples are retrieved.

        Returns
        -------
        List[ExperimentSample]
            The ExperimentSamples at the given indices.
        """
        if indices is None:
            # Return a list of the iterator over ExperimentData
            return list(self)

        return [self.get_experiment_sample(index) for index in indices]

    def _set_experiment_sample(self,
                               experiment_sample: ExperimentSample) -> None:
        """
        Sets the ExperimentSample at the given index.

        Parameters
        ----------
        experiment_sample : ExperimentSample
            The ExperimentSample to set.
        """
        for column, (value, is_disk) in experiment_sample._dict_output.items():

            if not self.domain.is_in_output(column):
                self.domain.add_output(column, to_disk=is_disk)

            self._output_data.set_data(
                index=experiment_sample.job_number, value=value,
                column=column)

        self._jobs.mark(experiment_sample._jobnumber, status=Status.FINISHED)

    @_access_file
    def _write_experiment_sample(self,
                                 experiment_sample: ExperimentSample) -> None:
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
        job_index = self._jobs.get_open_job()
        self._jobs.mark(job_index, status=Status.IN_PROGRESS)
        experiment_sample = self.get_experiment_sample(job_index)
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

    #                                                                      Jobs
    # =========================================================================

    def _set_error(self, index: int) -> None:
        """Mark the experiment_sample at the given index as error.

        Parameters
        ----------
        index
            index of the experiment_sample to mark as error
        """
        # self.jobs.mark_as_error(index)
        self._jobs.mark(index, status=Status.ERROR)
        self._output_data.set_data(
            index,
            value=['ERROR' for _ in self._output_data.names])

    @_access_file
    def _write_error(self, index: int):
        """Mark the experiment_sample at the given index as
         error and write to ExperimentData file.

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
        return self._jobs.is_all_finished()

    def mark(self, indices: Iterable[int],
             status: Literal['open', 'in progress', 'finished', 'error']):
        """Mark the jobs at the given indices with the given status.

        Parameters
        ----------
        indices : Iterable[int]
            indices of the jobs to mark
        status : Literal['open', 'in progress', 'finished', 'error']
            status to mark the jobs with: choose between: 'open', \
            'in progress', 'finished' or 'error'

        Raises
        ------
        ValueError
            If the given status is not any of 'open', 'in progress', \
            'finished' or 'error'
        """
        # Check if the status is in Status
        if not any(status.lower() == s.value for s in Status):
            raise ValueError(f"Invalid status {status} given. "
                             f"\nChoose from values: "
                             f"{', '.join([s.value for s in Status])}")

        self._jobs.mark(indices, status)

    def mark_all(self,
                 status: Literal['open', 'in progress', 'finished', 'error']):
        """Mark all the experiments with the given status

        Parameters
        ----------
        status : Literal['open', 'in progress', 'finished', 'error']
            status to mark the jobs with: \
            choose between:

            * 'open',
            * 'in progress',
            * 'finished'
            * 'error'

        Raises
        ------
        ValueError
            If the given status is not any of \
            'open', 'in progress', 'finished' or 'error'
        """
        self.mark(self._jobs.indices, status)

    def mark_all_error_open(self) -> None:
        """
        Mark all the experiments that have the status 'error' open
        """
        self._jobs.mark_all_error_open()

    def mark_all_in_progress_open(self) -> None:
        """
        Mark all the experiments that have the status 'in progress' open
        """
        self._jobs.mark_all_in_progress_open()

    def mark_all_nan_open(self) -> None:
        """
        Mark all the experiments that have 'nan' in output open
        """
        indices = self._output_data.get_index_with_nan()
        self.mark(indices=indices, status='open')
    #                                                            Datageneration
    # =========================================================================

    def evaluate(self, data_generator: DataGenerator,
                 mode: Literal['sequential', 'parallel',
                               'cluster', 'cluster_parallel'] = 'sequential',
                 kwargs: Optional[dict] = None,
                 output_names: Optional[List[str]] = None) -> None:
        """Run any function over the entirety of the experiments

        Parameters
        ----------
        data_generator : DataGenerator
            data generator to use
        mode : str, optional
            operational mode, by default 'sequential'. Choose between:

            * 'sequential' : Run the operation sequentially
            * 'parallel' : Run the operation on multiple cores
            * 'cluster' : Run the operation on the cluster
            * 'cluster_parallel' : Run the operation on the cluster in parallel

        kwargs, optional
            Any keyword arguments that need to
            be supplied to the function, by default None
        output_names : List[str], optional
            If you provide a function as data generator, you have to provide
            the names of all the output parameters that are in the return
            statement, in order of appearance.

        Raises
        ------
        ValueError
            Raised when invalid parallelization mode is specified
        """
        if kwargs is None:
            kwargs = {}

        if inspect.isfunction(data_generator):
            if output_names is None:
                raise TypeError(
                    ("If you provide a function as data generator, you have to"
                     "provide the names of the return arguments with the"
                     "output_names attribute."))
            data_generator = convert_function(
                f=data_generator, output=output_names)

        elif isinstance(data_generator, str):
            data_generator = _datagenerator_factory(
                data_generator, self.domain, kwargs)

        if mode.lower() == "sequential":
            return self._run_sequential(data_generator, kwargs)
        elif mode.lower() == "parallel":
            return self._run_multiprocessing(data_generator, kwargs)
        elif mode.lower() == "cluster":
            return self._run_cluster(data_generator, kwargs)
        elif mode.lower() == "cluster_parallel":
            return self._run_cluster_parallel(data_generator, kwargs)
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
                logger.debug(
                    f"Accessed experiment_sample \
                         {experiment_sample._jobnumber}")
            except NoOpenJobsError:
                logger.debug("No Open Jobs left")
                break

            try:

                # If kwargs is empty dict
                if not kwargs:
                    logger.debug(
                        f"Running experiment_sample "
                        f"{experiment_sample._jobnumber}")
                else:
                    logger.debug(
                        f"Running experiment_sample "
                        f"{experiment_sample._jobnumber} with kwargs {kwargs}")

                _experiment_sample = data_generator._run(
                    experiment_sample, **kwargs)  # no *args!
                self._set_experiment_sample(_experiment_sample)
            except Exception as e:
                error_msg = f"Error in experiment_sample \
                     {experiment_sample._jobnumber}: {e}"
                error_traceback = traceback.format_exc()
                logger.error(f"{error_msg}\n{error_traceback}")
                self._set_error(experiment_sample._jobnumber)

    def _run_multiprocessing(self, data_generator: DataGenerator,
                             kwargs: dict):
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

        def f(options: Dict[str, Any]) -> Tuple[ExperimentSample, int]:
            try:

                logger.debug(
                    f"Running experiment_sample "
                    f"{options['experiment_sample'].job_number}")

                return (data_generator._run(**options), 0)  # no *args!

            except Exception as e:
                error_msg = f"Error in experiment_sample \
                     {options['experiment_sample'].job_number}: {e}"
                error_traceback = traceback.format_exc()
                logger.error(f"{error_msg}\n{error_traceback}")
                return (options['experiment_sample'], 1)

        with mp.Pool() as pool:
            # maybe implement pool.starmap_async ?
            _experiment_samples: List[
                Tuple[ExperimentSample, int]] = pool.starmap(f, options)

        for _experiment_sample, exit_code in _experiment_samples:
            if exit_code == 0:
                self._set_experiment_sample(_experiment_sample)
            else:
                self._set_error(_experiment_sample.job_number)

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
            self = self.from_file(self.project_dir)
        except FileNotFoundError:  # If not found, store current
            self.store()

        while True:
            try:
                experiment_sample = self._get_open_job_data()
            except NoOpenJobsError:
                logger.debug("No Open jobs left!")
                break

            try:
                _experiment_sample = data_generator._run(
                    experiment_sample, **kwargs)
                self._write_experiment_sample(_experiment_sample)
            except Exception:
                n = experiment_sample.job_number
                error_msg = f"Error in experiment_sample {n}: "
                error_traceback = traceback.format_exc()
                logger.error(f"{error_msg}\n{error_traceback}")
                self._write_error(experiment_sample._jobnumber)
                continue

        self = self.from_file(self.project_dir)
        # Remove the lockfile from disk
        (self.project_dir / EXPERIMENTDATA_SUBFOLDER / LOCK_FILENAME
         ).with_suffix('.lock').unlink(missing_ok=True)

    def _run_cluster_parallel(
            self, data_generator: DataGenerator, kwargs: dict):
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
            self = self.from_file(self.project_dir)
        except FileNotFoundError:  # If not found, store current
            self.store()

        no_jobs = False

        while True:
            es_list = []
            for core in range(mp.cpu_count()):
                try:
                    es_list.append(self._get_open_job_data())
                except NoOpenJobsError:
                    logger.debug("No Open jobs left!")
                    no_jobs = True
                    break

            d = self.select([e.job_number for e in es_list])

            d._run_multiprocessing(
                data_generator=data_generator, kwargs=kwargs)

            # TODO access resource first!
            self.overwrite_disk(
                indices=d.index, input_data=d._input_data,
                output_data=d._output_data, jobs=d._jobs,
                domain=d.domain, add_if_not_exist=False)

            if no_jobs:
                break

        self = self.from_file(self.project_dir)
        # Remove the lockfile from disk
        (self.project_dir / EXPERIMENTDATA_SUBFOLDER / LOCK_FILENAME
         ).with_suffix('.lock').unlink(missing_ok=True)

    #                                                              Optimization
    # =========================================================================

    def optimize(self, optimizer: Optimizer | str,
                 data_generator: DataGenerator | str,
                 iterations: int,
                 kwargs: Optional[Dict[str, Any]] = None,
                 hyperparameters: Optional[Dict[str, Any]] = None,
                 x0_selection: Literal['best', 'random',
                                       'last',
                                       'new'] | ExperimentData = 'best',
                 sampler: Optional[Sampler | str] = 'random',
                 overwrite: bool = False,
                 callback: Optional[Callable] = None) -> None:
        """Optimize the experimentdata object

        Parameters
        ----------
        optimizer : Optimizer | str
            Optimizer object
        data_generator : DataGenerator | str
            DataGenerator object
        iterations : int
            number of iterations
        kwargs : Dict[str, Any], optional
            any additional keyword arguments that will be passed to
            the DataGenerator
        hyperparameters : Dict[str, Any], optional
            any additional keyword arguments that will be passed to
            the optimizer
        x0_selection : str | ExperimentData
            How to select the initial design. By default 'best'
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
        sampler: Sampler, optional
            If x0_selection = 'new', the sampler to use. By default 'random'
        overwrite: bool, optional
            If True, the optimizer will overwrite the current data. By default
            False
        callback : Callable, optional
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
        # Create the data generator object if a string reference is passed
        if isinstance(data_generator, str):
            data_generator: DataGenerator = _datagenerator_factory(
                data_generator=data_generator,
                domain=self.domain, kwargs=kwargs)

        # Create a copy of the optimizer object
        _optimizer = copy(optimizer)

        # Create the optimizer object if a string reference is passed
        if isinstance(_optimizer, str):
            _optimizer: Optimizer = _optimizer_factory(
                _optimizer, self.domain, hyperparameters)

        # Create the sampler object if a string reference is passed
        if isinstance(sampler, str):
            sampler: Sampler = _sampler_factory(sampler, self.domain)

        if _optimizer.type == 'scipy':
            self._iterate_scipy(
                optimizer=_optimizer, data_generator=data_generator,
                iterations=iterations, kwargs=kwargs,
                x0_selection=x0_selection,
                sampler=sampler,
                overwrite=overwrite,
                callback=callback)
        else:
            self._iterate(
                optimizer=_optimizer, data_generator=data_generator,
                iterations=iterations, kwargs=kwargs,
                x0_selection=x0_selection,
                sampler=sampler,
                overwrite=overwrite,
                callback=callback)

    def _iterate(self, optimizer: Optimizer, data_generator: DataGenerator,
                 iterations: int, kwargs: Dict[str, Any], x0_selection: str,
                 sampler: Sampler, overwrite: bool,
                 callback: Callable):
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
        last_index = self.index[-1] if not self.index.empty else -1

        if isinstance(x0_selection, str):
            if x0_selection == 'new':

                if iterations < optimizer._population:
                    raise ValueError(
                        f'For creating new samples, the total number of '
                        f'requested iterations ({iterations}) cannot be '
                        f'smaller than the population size '
                        f'({optimizer._population})')

                init_samples = ExperimentData.from_sampling(
                    domain=self.domain,
                    sampler=sampler,
                    n_samples=optimizer._population,
                    seed=optimizer._seed)

                init_samples.evaluate(
                    data_generator=data_generator, kwargs=kwargs,
                    mode='sequential')

                if callback is not None:
                    callback(init_samples)

                if overwrite:
                    _indices = init_samples.index + last_index + 1
                    self._overwrite_experiments(
                        experiment_sample=init_samples,
                        indices=_indices,
                        add_if_not_exist=True)

                else:
                    self.add_experiments(init_samples)

                x0_selection = 'last'
                iterations -= optimizer._population

        x0 = x0_factory(experiment_data=self, mode=x0_selection,
                        n_samples=optimizer._population)
        optimizer._set_data(x0)

        optimizer._check_number_of_datapoints()

        optimizer._construct_model(data_generator)

        for _ in range(number_of_updates(
                iterations,
                population=optimizer._population)):
            new_samples = optimizer.update_step(data_generator)

            # If new_samples is a tuple of input_data and output_data
            if isinstance(new_samples, tuple):
                new_samples = ExperimentData(
                    domain=self.domain,
                    input_data=new_samples[0],
                    output_data=new_samples[1],
                )
            # If applicable, evaluate the new designs:
            new_samples.evaluate(
                data_generator, mode='sequential', kwargs=kwargs)

            if callback is not None:
                callback(new_samples)

            if overwrite:
                _indices = new_samples.index + last_index + 1
                self._overwrite_experiments(experiment_sample=new_samples,
                                            indices=_indices,
                                            add_if_not_exist=True)

            else:
                self.add_experiments(new_samples)

            optimizer._set_data(self)

        if not overwrite:
            # Remove overiterations
            self.remove_rows_bottom(number_of_overiterations(
                iterations,
                population=optimizer._population))

        # Reset the optimizer
        # optimizer.reset(ExperimentData(domain=self.domain))

    def _iterate_scipy(self, optimizer: Optimizer,
                       data_generator: DataGenerator,
                       iterations: int, kwargs: dict,
                       x0_selection: str | ExperimentData,
                       sampler: Sampler, overwrite: bool,
                       callback: Callable):
        """Internal represenation of the iteration process for scipy-minimize
        optimizers.

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
        last_index = self.index[-1] if not self.index.empty else -1
        n_data_before_iterate = len(self)

        if isinstance(x0_selection, str):
            if x0_selection == 'new':

                if iterations < optimizer._population:
                    raise ValueError(
                        f'For creating new samples, the total number of '
                        f'requested iterations ({iterations}) cannot be '
                        f'smaller than the population size '
                        f'({optimizer._population})')

                init_samples = ExperimentData.from_sampling(
                    domain=self.domain,
                    sampler=sampler,
                    n_samples=optimizer._population,
                    seed=optimizer._seed)

                init_samples.evaluate(
                    data_generator=data_generator, kwargs=kwargs,
                    mode='sequential')

                if callback is not None:
                    callback(init_samples)

                if overwrite:
                    _indices = init_samples.index + last_index + 1
                    self._overwrite_experiments(
                        experiment_sample=init_samples,
                        indices=_indices,
                        add_if_not_exist=True)

                else:
                    self.add_experiments(init_samples)

                x0_selection = 'last'

        x0 = x0_factory(experiment_data=self, mode=x0_selection,
                        n_samples=optimizer._population)
        optimizer._set_data(x0)

        optimizer._check_number_of_datapoints()

        optimizer.run_algorithm(iterations, data_generator)

        new_samples: ExperimentData = optimizer.data.select(
            optimizer.data.index[1:])
        new_samples.evaluate(data_generator, mode='sequential', kwargs=kwargs)

        if callback is not None:
            callback(new_samples)

        if overwrite:
            self.add_experiments(
                optimizer.data.select([optimizer.data.index[-1]]))

        elif not overwrite:
            # Do not add the first element, as this is already
            # in the sampled data
            self.add_experiments(new_samples)

            # TODO: At the end, the data should have
            # n_data_before_iterate + iterations amount of elements!
            # If x_new is empty, repeat best x0 to fill up total iteration
            if len(self) == n_data_before_iterate:
                repeated_sample = self.get_n_best_output(
                    n_samples=1)

                for repetition in range(iterations):
                    self.add_experiments(repeated_sample)

            # Repeat last iteration to fill up total iteration
            if len(self) < n_data_before_iterate + iterations:
                last_design = self.get_experiment_sample(len(self)-1)

                while len(self) < n_data_before_iterate + iterations:
                    self.add_experiments(last_design)

        # Evaluate the function on the extra iterations
        self.evaluate(data_generator, mode='sequential', kwargs=kwargs)

        # Reset the optimizer
        # optimizer.reset(ExperimentData(domain=self.domain))

    #                                                                  Sampling
    # =========================================================================

    def sample(self, sampler: Sampler | SamplerNames, n_samples: int = 1,
               seed: Optional[int] = None, **kwargs) -> None:
        """Sample data from the domain providing the sampler strategy

        Parameters
        ----------
        sampler: Sampler | str
            Sampler callable or string of built-in sampler
            If a string is passed, it should be one of the built-in samplers:

            * 'random' : Random sampling
            * 'latin' : Latin Hypercube Sampling
            * 'sobol' : Sobol Sequence Sampling
            * 'grid' : Grid Search Sampling
        n_samples : int, optional
            Number of samples to generate, by default 1
        seed : Optional[int], optional
            Seed to use for the sampler, by default None

        Note
        ----
        When using the 'grid' sampler, an optional argument
        'stepsize_continuous_parameters' can be passed to specify the stepsize
        to cast continuous parameters to discrete parameters.

        - The stepsize should be a dictionary with the parameter names as keys\
        and the stepsize as values.
        - Alternatively, a single stepsize can be passed for all continuous\
        parameters.

        Raises
        ------
        ValueError
            Raised when invalid sampler type is specified
        """

        if isinstance(sampler, str):
            sampler = _sampler_factory(sampler, self.domain)

        sample_data: DataTypes = sampler(
            domain=self.domain, n_samples=n_samples, seed=seed, **kwargs)
        self.add(input_data=sample_data, domain=self.domain)

    #                                                         Project directory
    # =========================================================================

    def set_project_dir(self, project_dir: Path | str):
        """Set the directory of the f3dasm project folder.

        Parameters
        ----------
        project_dir : Path or str
            Path to the project directory
        """
        self.project_dir = _project_dir_factory(project_dir)


def x0_factory(experiment_data: ExperimentData,
               mode: str | ExperimentData, n_samples: int):
    """Set the initial population to the best n samples of the given data

    Parameters
    ----------
    experiment_data : ExperimentData
        Data to be used for the initial population
    mode : str
        Mode of selecting the initial population, by default 'best'
        The following modes are available:

            - best: select the best n samples
            - random: select n random samples
            - last: select the last n samples
    n_samples : int
        Number of samples to select

    Raises
    ------
    ValueError
        Raises when the mode is not recognized
    """
    if isinstance(mode, ExperimentData):
        x0 = mode

    elif mode == 'best':
        x0 = experiment_data.get_n_best_output(n_samples)

    elif mode == 'random':
        x0 = experiment_data.select(
            np.random.choice(
                experiment_data.index,
                size=n_samples, replace=False))

    elif mode == 'last':
        x0 = experiment_data.select(
            experiment_data.index[-n_samples:])

    else:
        raise ValueError(
            f'Unknown selection mode {mode}, use best, random or last')

    x0._reset_index()
    return x0
