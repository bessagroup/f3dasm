"""
The ExperimentData object is the main object used to store implementations
 of a design-of-experiments, keep track of results, perform optimization and
 extract data for machine learning purposes.
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
import functools
from collections import defaultdict
from copy import copy
from itertools import zip_longest
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

# Local
from ..datageneration import DataGenerator, _datagenerator_factory
from ..design import Domain, _domain_factory
from ..logger import logger
from ..optimization import Optimizer, _optimizer_factory
from ._io import (DOMAIN_FILENAME, EXPERIMENTDATA_SUBFOLDER,
                  INPUT_DATA_FILENAME, JOBS_FILENAME, LOCK_FILENAME, MAX_TRIES,
                  OUTPUT_DATA_FILENAME, _project_dir_factory)
from .experimentsample import ExperimentSample
from .samplers import Sampler, _sampler_factory
from .utils import DataTypes, deprecated

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class ExperimentData:
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

        _project_dir = _project_dir_factory(project_dir)
        _domain = _domain_factory(domain)

        # If input_data is a numpy array, create pd.Dataframe to include column
        # names from the domain
        if isinstance(input_data, np.ndarray):
            input_data = convert_numpy_to_dataframe_with_domain(
                array=input_data,
                names=_domain.names)

        # Same with output data
        if isinstance(output_data, np.ndarray):
            output_data = convert_numpy_to_dataframe_with_domain(
                array=output_data,
                names=_domain.output_names)

        _data = data_factory(
            input_data=input_data, output_data=output_data)

        # If the domain is None, try to infer it from the input_data and output
        # data
        if not _domain and _data:
            _domain = infer_domain_from_data(_data)

        # if jobs is not None, overwrite the job status
        if jobs is not None:
            ...

        self.data = _data
        self.domain = _domain
        self.project_dir = _project_dir

    def __len__(self):
        return len(self.data)

    def __iter__(self) -> Iterator[Tuple[int, ExperimentSample]]:
        return iter(self.data.items())

    def __add__(self, __o: ExperimentData) -> ExperimentData:
        # copy and reset self
        copy_self = copy(self).reset_index()
        copy_self._add(__o)
        return copy_self

    def __eq__(self, __o: ExperimentData) -> bool:
        return self.data == __o.data and self.domain == __o.domain

    def __getitem__(self, key: int | Iterable[int]) -> ExperimentData:
        if isinstance(key, int):
            key = [key]

        if not pd.Index(key).isin(self.index).all():
            raise KeyError(f"Keys {key} not found in index")

        return ExperimentData.from_data(data={k: self.data[k] for k in key},
                                        domain=self.domain)

    def _repr_html_(self) -> str:
        return self.to_multiindex()._repr_html_()

    def __repr__(self) -> str:
        return self.to_multiindex().__repr__()

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
        @functools.wraps(operation)
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
        return pd.Index(self.data.keys())

    #                                                  Alternative constructors
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
            return _from_file_attempt(project_dir)
        except FileNotFoundError:
            try:
                filename_with_path = Path(get_original_cwd()) / project_dir
            except ValueError:  # get_original_cwd() hydra initialization error
                raise FileNotFoundError(
                    f"Cannot find the folder {project_dir} !")

            return _from_file_attempt(filename_with_path)

    @classmethod
    def from_sampling(cls, sampler: Sampler | str, domain: Domain | DictConfig,
                      **kwargs):
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
        data = cls(domain=domain)
        data.sample(sampler=sampler, **kwargs)
        return data

    @classmethod
    def from_yaml(cls, config: DictConfig) -> ExperimentData:
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
    def from_data(cls, data: Optional[Dict[int, ExperimentSample]] = None,
                  domain: Optional[Domain] = None,
                  project_dir: Optional[Path] = None) -> ExperimentData:

        if data is None:
            data = {}

        if domain is None:
            domain = Domain()

        experiment_data = cls()

        experiment_data.data = defaultdict(ExperimentSample, data)
        experiment_data.domain = domain
        experiment_data.project_dir = _project_dir_factory(project_dir)
        return experiment_data

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
        return self[indices]

    # Not used
    @deprecated(version="2.0.0")
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
        ...

    def select_with_status(
            self,
            status: Literal['open', 'in_progress', 'finished', 'error']
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
        idx = [i for i, es in self if es.is_status(status)]
        return self[idx]

    # Do you need to use this?
    @deprecated(version="2.0.0")
    def get_input_data(
            self,
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
        ...

    # Do you need to use this?
    @deprecated(version="2.0.0")
    def get_output_data(
        self,
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
        ...

    #                                                                    Export
    # =========================================================================

    def store(self, project_dir: Optional[Path | str] = None):
        """Write the ExperimentData to disk in the project directory.

        Parameters
        ----------
        project_dir : Optional[Path | str], optional
            The f3dasm project directory to store the
            ExperimentData object to, by default None.

        Note
        ----
        If no project directory is provided, the ExperimentData object is
        stored in the directory provided by the `.project_dir` attribute that
        is set upon creation of the object.

        The ExperimentData object is stored in a subfolder 'experiment_data'.

        The ExperimentData object is stored in four files:

        * the input data (`input.csv`)
        * the output data (`output.csv`)
        * the jobs (`jobs.pkl`)
        * the domain (`domain.pkl`)

        To avoid the ExperimentData to be written simultaneously by multiple
        processes, a '.lock' file is automatically created
        in the project directory. Concurrent process can only sequentially
        access the lock file. This lock file is removed after the
        ExperimentData object is written to disk.
        """
        ...

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert the ExperimentData object to a tuple of numpy arrays.

        Returns
        -------
        tuple
            A tuple containing two numpy arrays,
            the first one for input columns,
            and the second for output columns.
        """
        df_input, df_output = self.to_pandas()
        return df_input.to_numpy(), df_output.to_numpy()

    def to_pandas(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Convert the ExperimentData object to a pandas DataFrame.

        Returns
        -------
        tuple
            A tuple containing two pandas DataFrames,
            the first one for input columns, and the second for output
        """
        return (
            pd.DataFrame([es.input_data for _, es in self], index=self.index),
            pd.DataFrame([es.output_data for _, es in self], index=self.index)
        )

    def to_xarray(self) -> xr.Dataset:
        """
        Convert the ExperimentData object to an xarray Dataset.

        Returns
        -------
        xarray.Dataset
            An xarray Dataset containing the data.
        """
        ...
        # df_input, df_output = self.to_pandas()
        # da_input = xr.DataArray(df_input, dims=['iterations', 'input'],
        #                         coords={'iterations': self.index,
        #                                 'input': df_input.columns})
        # da_output = xr.DataArray(df_output, dims=['iterations', 'output'],
        #                          coords={'iterations': self.index,
        #                                  'output': df_output.columns})

        # return xr.Dataset(
        #     {'input': da_input,
        #      'output': da_output})

    def get_n_best_output(self, n_samples: int) -> ExperimentData:
        """Get the n best samples from the output data.
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

        The n best samples are selected based on the output data.
        The output data is sorted based on the first output parameter.
        The n best samples are selected based on this sorting.
        """
        ...

    def to_multiindex(self) -> pd.DataFrame:
        list_of_dicts = [sample.to_multiindex() for _, sample in self]
        return pd.DataFrame(merge_dicts(list_of_dicts), index=self.index)

    #                                                     Append or remove data
    # =========================================================================

    # Not used
    @deprecated(version="2.0.0")
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
        ...

    def add_experiments(self,
                        data: ExperimentSample | ExperimentData
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
        if isinstance(data, ExperimentSample):
            self._add_experiment_sample(data)

        elif isinstance(data, ExperimentData):
            self._add(data)

        else:
            raise ValueError((
                f"The input to this function should be an ExperimentSample or "
                f"ExperimentData object, not {type(data)} ")
            )
        ...

    # Not used
    @deprecated(version="2.0.0")
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
        ...

    # Not used
    @deprecated(version="2.0.0")
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
        ...

    # Used in parallel mode
    @deprecated(version="2.0.0")
    def overwrite_disk(
        self, indices: Iterable[int],
            domain: Optional[Domain] = None,
            input_data: Optional[DataTypes] = None,
            output_data: Optional[DataTypes] = None,
            jobs: Optional[Path | str] = None,
            add_if_not_exist: bool = False
    ) -> None:
        ...

    # Not used
    @deprecated(version="2.0.0")
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
        ...

    # Not used
    @deprecated(version="2.0.0")
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
        ...

    def remove_rows_bottom(self, number_of_rows: int):
        """
        Remove a number of rows from the end of the ExperimentData object.

        Parameters
        ----------
        number_of_rows : int
            Number of rows to remove from the bottom.
        """
        # remove the last n rows
        for i in range(number_of_rows):
            self.data.pop(self.index[-1])

    # Not used
    @deprecated(version="2.0.0", message="Use reset_index() instead.")
    def _reset_index(self) -> None:
        """
        Reset the index of the ExperimentData object.
        """
        ...

    def reset_index(self) -> ExperimentData:
        """
        Reset the index of the ExperimentData object.

        Returns
        -------
        ExperimentData
            ExperimentData object with a reset index.
        """
        return ExperimentData.from_data(
            data={i: v for i, v in enumerate(self.data.values())},
            domain=self.domain,
            project_dir=self.project_dir)

    def join(self, experiment_data: ExperimentData) -> ExperimentData:
        """Join two ExperimentData objects.

        Parameters
        ----------
        experiment_data : ExperimentData
            The other ExperimentData object to join with.

        Returns
        -------
        ExperimentData
            The joined ExperimentData object.
        """
        copy_self = self.reset_index()
        # TODO: Reset isnt necessary, only copy
        copy_other = experiment_data.reset_index()

        for (i, es_self), (_, es_other) in zip(copy_self, copy_other):
            copy_self.data[i] = es_self + es_other

        copy_self.domain += copy_other.domain

        return copy_self

    def _add(self, experiment_data: ExperimentData):
        # copy and reset self
        copy_other = experiment_data.reset_index()

        # Find the last key in my_dict
        last_key = max(self.index) if self else -1

        # Update keys of other dict
        other_updated_data = {
            last_key + 1 + i: v for i, v in enumerate(
                copy_other.data.values())}

        self.data.update(other_updated_data)
        self.domain += copy_other.domain

    def _add_experiment_sample(self, experiment_sample: ExperimentSample):
        last_key = max(self.index) if self else -1
        self.data[last_key + 1] = experiment_sample

    def _overwrite(self, experiment_data: ExperimentData,
                   indices: Iterable[int],
                   add_if_not_exist: bool = False):
        if len(indices) != len(experiment_data):
            raise ValueError((
                f"The number of indices ({len(indices)}) must match the number"
                f"of experiments ({len(experiment_data)}).")
            )
        copy_other = experiment_data.reset_index()

        for (_, es), id in zip(copy_other, indices):
            self.data[id] = es

        self.domain += copy_other.domain

    #                                                          ExperimentSample
    # =========================================================================

    def get_experiment_sample(self, id: int) -> ExperimentSample:
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
        return self.data[id]

    @deprecated(version="2.0.0")
    def get_experiment_samples(
            self, indices: Iterable[int]) -> List[ExperimentSample]:
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
        return [self.data[i] for i in indices]

    def _set_experiment_sample(
            self, experiment_sample: ExperimentSample, id: int):
        """
        Sets the ExperimentSample at the given index.

        Parameters
        ----------
        experiment_sample : ExperimentSample
            The ExperimentSample to set.
        """
        self.data[id] = experiment_sample

        # TODO: Is this the best way to do this? Cant we do this with
        # The key-value pairs and the Domain only ?
        # Check the keys that are registered and add them to the output data
        # TODO: Fix that the to_disk parameter is also retrieved
        for name, to_disk in experiment_sample.registered_keys.items():
            self.domain.add_output(name=name, to_disk=to_disk, exist_ok=True)

        experiment_sample.clean_registered_keys()

    # Used in parallel mode
    @deprecated(version="2.0.0")
    def _write_experiment_sample(self,
                                 experiment_sample: ExperimentSample) -> None:
        """
        Sets the ExperimentSample at the given index.

        Parameters
        ----------
        experiment_sample : ExperimentSample
            The ExperimentSample to set.
        """
        ...

    # Used in parallel mode
    def _access_open_job_data(self) -> Tuple[int, ExperimentSample]:
        """Get the data of the first available open job.

        Returns
        -------
        ExperimentSample
            The ExperimentSample object of the first available open job.
        """
        return self.get_open_job()

    @deprecated(version="2.0.0")
    def _get_open_job_data(self) -> ExperimentSample:
        """Get the data of the first available open job by
        accessing the ExperimenData on disk.

        Returns
        -------
        ExperimentSample
            The ExperimentSample object of the first available open job.
        """
        ...

    def get_open_job(self) -> Tuple[int, ExperimentSample]:
        for id, es in self:
            if es.is_status('open'):
                es.mark('in_progress')
                return id, es

        return None, ExperimentSample()

    #                                                                      Jobs
    # =========================================================================

    def is_all_finished(self) -> bool:
        """Check if all jobs are finished

        Returns
        -------
        bool
            True if all jobs are finished, False otherwise
        """
        return all(es.is_status('finished') for _, es in self)

    def mark(self, indices: int | Iterable[int],
             status: Literal['open', 'in_progress', 'finished', 'error']):
        """Mark the jobs at the given indices with the given status.

        Parameters
        ----------
        indices : Iterable[int]
            indices of the jobs to mark
        status : Literal['open', 'in progress', 'finished', 'error']
            status to mark the jobs with: choose between: 'open',
            'in progress', 'finished' or 'error'

        Raises
        ------
        ValueError
            If the given status is not any of 'open', 'in progress',
            'finished' or 'error'
        """
        if isinstance(indices, int):
            indices = [indices]
        for i in indices:
            self.data[i].mark(status)

    def mark_all(self,
                 status: Literal['open', 'in_progress', 'finished', 'error']):
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
        for _, es in self:
            es.mark(status)

    @deprecated(version="2.0.0")
    def mark_all_error_open(self) -> None:
        """
        Mark all the experiments that have the status 'error' open
        """
        ...

    @deprecated(version="2.0.0")
    def mark_all_in_progress_open(self) -> None:
        """
        Mark all the experiments that have the status 'in progress' open
        """
        ...

    @deprecated(version="2.0.0")
    def mark_all_nan_open(self) -> None:
        """
        Mark all the experiments that have 'nan' in output open
        """
        ...

    #                                                            Datageneration
    # =========================================================================

    def evaluate(self, data_generator: DataGenerator,
                 mode: Literal['sequential', 'parallel',
                               'cluster', 'cluster_parallel'] = 'sequential',
                 output_names: Optional[List[str]] = None,
                 **kwargs) -> None:
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

        output_names : List[str], optional
            If you provide a function as data generator, you have to provide
            the names of all the output parameters that are in the return
            statement, in order of appearance.

        Raises
        ------
        ValueError
            Raised when invalid parallelization mode is specified

        Notes
        -----
        Any additional keyword arguments are passed to the data generator.
        """

        # Create
        data_generator = _datagenerator_factory(
            data_generator=data_generator, output_names=output_names, **kwargs)

        # Initialize
        data_generator.init(data=self)

        # Call
        self = data_generator.call(mode=mode, **kwargs)

    # # It would be cool to have a different variant of ExperimentData that
    # # has disk reading/writing operations for the data

    #                                                              Optimization
    # =========================================================================

    def optimize(self, optimizer: Optimizer | str | Callable,
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
        optimizer : Optimizer | str | Callable
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
        if kwargs is None:
            kwargs = {}

        # Create the data generator object if a string reference is passed
        if isinstance(data_generator, str):
            data_generator: DataGenerator = _datagenerator_factory(
                data_generator=data_generator, **kwargs)

        # # Create a copy of the optimizer object
        # _optimizer = copy(optimizer)

        if hyperparameters is None:
            hyperparameters = {}

        # Create the optimizer object if a string reference is passed
        if isinstance(optimizer, str):
            optimizer: Optimizer = _optimizer_factory(
                optimizer=optimizer, **hyperparameters)

        # Create the sampler object if a string reference is passed
        if isinstance(sampler, str):
            sampler: Sampler = _sampler_factory(sampler=sampler)

        last_index = self.index[-1] if not self.index.empty else -1

        if isinstance(x0_selection, str):
            if x0_selection == 'new':

                if iterations < optimizer._population:
                    raise ValueError(
                        f'For creating new samples, the total number of '
                        f'requested iterations ({iterations}) cannot be '
                        f'smaller than the population size '
                        f'({optimizer._population})')

                init_samples = ExperimentData(domain=self.domain)
                init_samples.remove_rows_bottom(len(init_samples))

                init_samples.sample(sampler=sampler,
                                    n_samples=optimizer._population,
                                    seed=optimizer._seed)

                init_samples.evaluate(
                    data_generator=data_generator, mode='sequential',
                    **kwargs)

                if callback is not None:
                    callback(init_samples)

                if overwrite:
                    _indices = init_samples.index + last_index + 1
                    self._overwrite(
                        experiment_data=init_samples,
                        indices=_indices,
                        add_if_not_exist=True)

                else:
                    self._add(init_samples)

                x0_selection = 'last'
                iterations -= optimizer._population

        x0 = x0_factory(experiment_data=self, mode=x0_selection,
                        n_samples=optimizer._population)

        x0.evaluate(data_generator=data_generator, mode='sequential',
                    **kwargs)

        optimizer.init(data=x0, data_generator=data_generator)
        self._add(optimizer.call(iterations=iterations,
                                 kwargs=kwargs,
                                 x0_selection=x0_selection,
                                 sampler=sampler,
                                 overwrite=overwrite,
                                 callback=callback,
                                 last_index=last_index)
                  )

    #                                                                  Sampling
    # =========================================================================

    def sample(self, sampler: Sampler | str, **kwargs) -> None:
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

        # Creation
        sampler = _sampler_factory(sampler=sampler, **kwargs)

        # Initialization
        sampler.init(domain=self.domain)

        # Sampling
        sample_data: DataTypes = sampler.sample(**kwargs)

        # Adding samples to the ExperimentData object
        self._add(ExperimentData(domain=self.domain,
                                 input_data=sample_data))

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

    def remove_lockfile(self):
        """
        Remove the lock file from the project directory

        Note
        ----
        The lock file is automatically created when the ExperimentData object
        is written to disk. Concurrent processes can only sequentially access
        the lock file. This lock file is removed after the ExperimentData
        object is written to disk.
        """
        (self.project_dir / EXPERIMENTDATA_SUBFOLDER / LOCK_FILENAME
         ).with_suffix('.lock').unlink(missing_ok=True)


# =============================================================================


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

    return x0.reset_index()


def _from_file_attempt(project_dir: Path) -> ExperimentData:
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
        return ExperimentData(domain=subdirectory / DOMAIN_FILENAME,
                              input_data=subdirectory / INPUT_DATA_FILENAME,
                              output_data=subdirectory / OUTPUT_DATA_FILENAME,
                              jobs=subdirectory / JOBS_FILENAME,
                              project_dir=project_dir)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Cannot find the files from {subdirectory}.")


def convert_numpy_to_dataframe_with_domain(
        array: np.ndarray, names: Optional[Domain]) -> pd.DataFrame:
    if not names:
        names = [f'{i}' for i in range(array.shape[1])]

    return pd.DataFrame(array, columns=names)


def infer_domain_from_data(data: Dict[int, ExperimentSample]) -> Domain:
    df_input = pd.DataFrame(
        [es.input_data for _, es in data.items()], index=data.keys())
    df_output = pd.DataFrame(
        [es.output_data for _, es in data.items()], index=data.keys())
    return Domain.from_dataframe(df_input=df_input, df_output=df_output)


def _dict_from_numpy(arr: np.ndarray,
                     names: Optional[Iterable[str]]) -> Dict[str, Any]:
    assert arr.ndim == 1, "Array must be 1D"

    names = names if names is not None else [
        f'x{i}' for i in range(arr.shape[0])]
    return dict(zip(names, arr))


def merge_dicts(list_of_dicts):
    merged_dict = defaultdict(list)

    # Get all unique keys from all dictionaries
    all_keys = sorted({key for d in list_of_dicts for key in d})

    # Define the desired order for the first element of the tuple
    order = {'jobs': 0, 'input': 1, 'output': 2}

    # Sort the keys first by the defined order then alphabetically within
    # each group
    sorted_keys = sorted(all_keys, key=lambda k: (
        order.get(k[0], float('inf')), k))

    # Iterate over each dictionary and insert None for missing keys
    for d in list_of_dicts:
        for key in sorted_keys:
            # Use None for missing keys
            merged_dict[key].append(d.get(key, None))

    return dict(merged_dict)


def _dict_factory(data: Optional[DataTypes]) -> List[Dict[str, Any]]:
    """
    Convert the DataTypes to a list of dictionaries

    Parameters
    ----------
    data : Optional[DataTypes]
        The data to be converted

    Returns
    -------
    List[Dict[str, Any]]
        The converted data as a list of dictionaries

    Raises
    ------
    ValueError
        Raised when the data type is not supported

    Notes
    -----
    If the data is None, an empty list is returned.
    """
    if data is None:
        return []

    # check if data is already a list of dicts
    elif isinstance(data, list) and all(isinstance(d, dict) for d in data):
        return data

    # This one is not reached since numpy is already handled earlier
    elif isinstance(data, np.ndarray):
        return [_dict_from_numpy(d) for d in data]

    # If the data is a pandas DataFrame, convert it to a list of dictionaries
    elif isinstance(data, pd.DataFrame):
        return [row._asdict() for row in data.itertuples(index=False)]

    # TODO: Add support for str argument

    raise ValueError(f"Data type {type(data)} not supported")


def data_factory(input_data: Optional[DataTypes],
                 output_data: Optional[DataTypes]
                 ) -> Dict[int, ExperimentSample]:
    """
    Convert the input and output data to a defaultdictionary
    of ExperimentSamples

    Parameters
    ----------
    input_data : Optional[DataTypes]
        The input data to be converted

    output_data : Optional[DataTypes]
        The output data to be converted

    Returns
    -------
    Dict[int, ExperimentSample]
        The converted data as a defaultdict of ExperimentSamples

    """
    # Create two lists of dictionaries from the input data
    _input_data: List[Dict[str, Any]] = _dict_factory(input_data)
    _output_data: List[Dict[str, Any]] = _dict_factory(output_data)

    # Combine the two lists into a dictionary of ExperimentSamples
    data = {index: ExperimentSample(input_data=input_data,
                                    output_data=output_data)
            for index, (input_data, output_data) in enumerate(
                zip_longest(_input_data, _output_data))}

    return defaultdict(ExperimentSample, data)
