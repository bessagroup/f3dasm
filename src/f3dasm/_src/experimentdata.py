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
from functools import partial
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

from ._io import (DOMAIN_FILENAME, EXPERIMENTDATA_SUBFOLDER,
                  INPUT_DATA_FILENAME, JOBS_FILENAME, LOCK_FILENAME, MAX_TRIES,
                  OUTPUT_DATA_FILENAME, _project_dir_factory, store_to_disk)
# Local
from .core import Block, DataGenerator
from .datageneration import _datagenerator_factory
from .design import Domain, _domain_factory, _sampler_factory
from .experimentsample import ExperimentSample
from .logger import logger
from .optimization import _optimizer_factory

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class ExperimentData:
    def __init__(
        self,
            domain: Optional[Domain] = None,
            input_data: Optional[
                pd.DataFrame | np.ndarray
                | List[Dict[str, Any]] | str | Path] = None,
            output_data: Optional[
                pd.DataFrame | np.ndarray
                | List[Dict[str, Any]] | str | Path] = None,
            jobs: Optional[pd.Series] = None,
            project_dir: Optional[Path] = None):
        """
        Initializes an instance of ExperimentData.

        Parameters
        ----------
        domain : Domain, optional
            The domain of the experiment, by default None
        input_data : pd.DataFrame | np.ndarray | List[Dict[str, Any]]
          | Path | str, optional
            The input data of the experiment, by default None
        output_data : pd.DataFrame | np.ndarray | List[Dict[str, Any]]
          | Path | str, optional
            The output data of the experiment, by default None
        jobs : pandas.Series, optional
            The status of all the jobs, by default None
        project_dir : Path | str, optional
            A user-defined directory where the f3dasm project folder will be
            created, by default the current working directory.

        Note
        ----

        The following data formats are supported for input and output data:

        * numpy array
        * pandas Dataframe
        * path to a csv file

        If no domain object is provided, the domain is inferred from the
        input_data.

        If the provided project_dir does not exist, it will be created.

        Raises
        ------

        ValueError
            If the input_data is a numpy array, the domain has to be provided.
        """
        _domain = _domain_factory(domain)
        _project_dir = _project_dir_factory(project_dir)
        _jobs = jobs_factory(jobs)

        # If input_data is a numpy array, create pd.Dataframe to include column
        # names from the domain
        if isinstance(input_data, np.ndarray):
            input_data = convert_numpy_to_dataframe_with_domain(
                array=input_data,
                names=_domain.input_names,
                mode='input')

        # Same with output data
        if isinstance(output_data, np.ndarray):
            output_data = convert_numpy_to_dataframe_with_domain(
                array=output_data,
                names=_domain.output_names,
                mode='output')

        _input_data = _dict_factory(data=input_data)
        _output_data = _dict_factory(data=output_data)

        # If the domain is empty, try to infer it from the input_data
        # and output_data
        if not _domain:
            _domain = Domain.from_data(input_data=_input_data,
                                       output_data=_output_data)

        _data = data_factory(
            input_data=_input_data, output_data=_output_data,
            domain=_domain, jobs=_jobs, project_dir=_project_dir
        )

        self.data = _data
        self.domain = _domain
        self.project_dir = _project_dir

        # Store to_disk objects so that the references are kept only
        for id, experiment_sample in self:
            self.store_experimentsample(experiment_sample, id)

    def __len__(self):
        """
        Returns the number of experiments in the ExperimentData object.

        Returns
        -------
        int
            Number of experiments.

        Examples
        --------
        >>> experimentdata = ExperimentData(input_data=np.array([1, 2, 3]),)
        >>> len(experiment_data)
        3
        """
        return len(self.data)

    def __iter__(self) -> Iterator[Tuple[int, ExperimentSample]]:
        """
        Returns an iterator over the ExperimentData object.

        Returns
        -------
        Iterator[Tuple[int, ExperimentSample]]
            Iterator over the ExperimentData object.

        Examples
        --------
        >>> for id, sample in experiment_data:
        ...     print(id, sample)
        0 ExperimentSample(...)
        """
        return iter(self.data.items())

    def __add__(self, __o: ExperimentData) -> ExperimentData:
        """
        Adds two ExperimentData objects.

        Parameters
        ----------
        __o : ExperimentData
            The other ExperimentData object to add.

        Returns
        -------
        ExperimentData
            The combined ExperimentData object.

        Examples
        --------
        >>> combined_data = experiment_data1 + experiment_data2
        """
        copy_self = copy(self).reset_index()
        copy_self._add(__o)
        return copy_self

    def __eq__(self, __o: ExperimentData) -> bool:
        """
        Checks if two ExperimentData objects are equal.

        Parameters
        ----------
        __o : ExperimentData
            The other ExperimentData object to compare.

        Returns
        -------
        bool
            True if the objects are equal, False otherwise.

        Notes
        -----
        Two ExperimentData objects are considered equal if their data, domain
        and project_dir are equal.

        Examples
        --------
        >>> experiment_data1 == experiment_data2
        True
        """
        return (self.data == __o.data and self.domain == __o.domain
                and self.project_dir == __o.project_dir)

    def __getitem__(self, key: int | Iterable[int]) -> ExperimentData:
        """
        Gets a subset of the ExperimentData object.

        Parameters
        ----------
        key : int or Iterable[int]
            The indices to select.

        Returns
        -------
        ExperimentData
            The selected subset of the ExperimentData object.

        """
        if isinstance(key, int):
            key = [key]

        return ExperimentData.from_data(
            data={k: self.data[k] for k in self.index[key]},
            domain=self.domain,
            project_dir=self.project_dir)

    def _repr_html_(self) -> str:
        """
        Returns an HTML representation of the ExperimentData object.

        Returns
        -------
        str
            HTML representation of the ExperimentData object.

        Examples
        --------
        >>> experiment_data._repr_html_()
        '<div>...</div>'
        """
        return self.to_multiindex()._repr_html_()

    def __repr__(self) -> str:
        """
        Returns a string representation of the ExperimentData object.

        Returns
        -------
        str
            String representation of the ExperimentData object.

        Examples
        --------
        >>> repr(experiment_data)
        'ExperimentData(...)'
        """
        return self.to_multiindex().__repr__()

    def access_file(self, operation: Callable) -> Callable:
        """
        Wrapper for accessing a single resource with a file lock.

        Parameters
        ----------
        operation : Callable
            The operation to be performed on the resource.

        Returns
        -------
        Callable
            The wrapped operation.

        Examples
        --------
        >>> @experiment_data.access_file
        ... def read_data(project_dir):
        ...     # read data from file
        ...     pass
        """
        @functools.wraps(operation)
        def wrapper_func(project_dir: Path, *args, **kwargs) -> None:
            lock = FileLock(
                (project_dir / EXPERIMENTDATA_SUBFOLDER / LOCK_FILENAME
                 ).with_suffix('.lock'))

            # If the lock has been acquired:
            with lock:
                tries = 0
                while tries < MAX_TRIES:
                    # try:
                    #     print(f"{args=}, {kwargs=}")
                    #     self = ExperimentData.from_file(project_dir)
                    #     value = operation(*args, **kwargs)
                    #     self.store()
                    #     break
                    try:
                        # Load a fresh instance of ExperimentData from file
                        loaded_self = ExperimentData.from_file(
                            self.project_dir)

                        # Call the operation with the loaded instance
                        # Replace the self in args with the loaded instance
                        # Modify the first argument
                        args = (loaded_self,) + args[1:]
                        value = operation(*args, **kwargs)
                        loaded_self.store()
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

        return partial(wrapper_func, project_dir=self.project_dir)

    #                                                                Properties
    # =========================================================================

    @property
    def index(self) -> pd.Index:
        """
        Returns an iterable of the job number of the experiments.

        Returns
        -------
        pd.Index
            The job number of all the experiments in pandas Index format.

        Examples
        --------
        >>> experiment_data.index
        Int64Index([0, 1, 2], dtype='int64')
        """
        return pd.Index(self.data.keys())

    @property
    def jobs(self) -> pd.Series:
        """
        Returns the status of all the jobs.

        Returns
        -------
        pd.Series
            The status of all the jobs.

        Examples
        --------
        >>> experiment_data.jobs
        0    open
        1    finished
        dtype: object
        """
        return pd.Series({id: es.job_status.name for id, es in self})

    #                                                  Alternative constructors
    # =========================================================================

    @classmethod
    def from_file(cls: Type[ExperimentData],
                  project_dir: Path | str) -> ExperimentData:
        """
        Create an ExperimentData object from .csv and .json files.

        Parameters
        ----------
        project_dir : Path or str
            User defined path of the experimentdata directory.

        Returns
        -------
        ExperimentData
            ExperimentData object containing the loaded data.

        Examples
        --------
        >>> experiment_data = ExperimentData.from_file('path/to/project_dir')
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
    def from_sampling(cls, sampler: Block | str,
                      domain: Domain | DictConfig | str | Path,
                      **kwargs):
        """
        Create an ExperimentData object from a sampler.

        Parameters
        ----------
        sampler : Block or str
            Sampler object containing the sampling strategy or one of the
            built-in sampler names.
        domain : Domain or DictConfig
            Domain object containing the domain of the experiment or hydra
            DictConfig object containing the configuration.
        **kwargs
            Additional keyword arguments passed to the sampler.

        Returns
        -------
        ExperimentData
            ExperimentData object containing the sampled data.

        Examples
        --------
        >>> experiment_data = ExperimentData.from_sampling('random', domain)
        """
        data = cls(domain=domain)
        data.sample(sampler=sampler, **kwargs)
        return data

    @classmethod
    def from_yaml(cls, config: DictConfig) -> ExperimentData:
        """
        Create an ExperimentData object from a YAML configuration.

        Parameters
        ----------
        config : DictConfig
            Hydra DictConfig object containing the configuration.

        Returns
        -------
        ExperimentData
            ExperimentData object containing the loaded data.

        Examples
        --------
        >>> experiment_data = ExperimentData.from_yaml(config)
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
    def from_data(cls, data: Optional[Dict[int, ExperimentSample]] = None,
                  domain: Optional[Domain] = None,
                  project_dir: Optional[Path] = None) -> ExperimentData:
        """
        Create an ExperimentData object from existing data.

        Parameters
        ----------
        data : dict of int to ExperimentSample, optional
            The existing data, by default None.
        domain : Domain, optional
            The domain of the data, by default None.
        project_dir : Path, optional
            The project directory, by default None.

        Returns
        -------
        ExperimentData
            ExperimentData object containing the loaded data.

        Examples
        --------
        >>> experiment_data = ExperimentData.from_data(data, domain)
        """
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
        """
        Select a subset of the ExperimentData object.

        Parameters
        ----------
        indices : int or Iterable[int]
            The indices to select.

        Returns
        -------
        ExperimentData
            The selected subset of the ExperimentData object.

        Examples
        --------
        >>> subset = experiment_data.select([0, 1, 2])
        """
        return self[indices]

    def select_with_status(
            self,
            status: Literal['open', 'in_progress', 'finished', 'error']
    ) -> ExperimentData:
        """
        Select a subset of the ExperimentData object with a given status.

        Parameters
        ----------
        status : {'open', 'in_progress', 'finished', 'error'}
            The status to select.

        Returns
        -------
        ExperimentData
            The selected subset of the ExperimentData object with the given
            status.

        Examples
        --------
        >>> subset = experiment_data.select_with_status('finished')
        """
        idx = [i for i, es in self if es.is_status(status)]
        return self[idx]

    #                                                                    Export
    # =========================================================================

    def store(self, project_dir: Optional[Path | str] = None):
        """
        Write the ExperimentData to disk in the project directory.

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
        * the jobs (`jobs.csv`)
        * the domain (`domain.pkl`)

        To avoid the ExperimentData to be written simultaneously by multiple
        processes, a '.lock' file is automatically created
        in the project directory. Concurrent process can only sequentially
        access the lock file. This lock file is removed after the
        ExperimentData object is written to disk.
        """
        if project_dir is not None:
            self.set_project_dir(project_dir)

        subdirectory = self.project_dir / EXPERIMENTDATA_SUBFOLDER

        # Create the experimentdata subfolder if it does not exist
        subdirectory.mkdir(parents=True, exist_ok=True)

        df_input, df_output = self.to_pandas(keep_references=True)

        df_input.to_csv(
            (subdirectory / INPUT_DATA_FILENAME).with_suffix('.csv'))
        df_output.to_csv(
            (subdirectory / OUTPUT_DATA_FILENAME).with_suffix('.csv'))
        self.domain.store(subdirectory / DOMAIN_FILENAME)
        self.jobs.to_csv((subdirectory / JOBS_FILENAME).with_suffix('.csv'))

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert the ExperimentData object to a tuple of numpy arrays.

        Returns
        -------
        tuple of np.ndarray
            A tuple containing two numpy arrays, the first one for input
            columns, and the second for output columns.

        Examples
        --------
        >>> input_array, output_array = experiment_data.to_numpy()
        """
        df_input, df_output = self.to_pandas(keep_references=False)
        return df_input.to_numpy(), df_output.to_numpy()

    def to_pandas(self, keep_references: bool = False
                  ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Convert the ExperimentData object to pandas DataFrames.

        Parameters
        ----------
        keep_references : bool, optional
            If True, the references to the output data are kept, by default
            False.

        Returns
        -------
        tuple of pd.DataFrame
            A tuple containing two pandas DataFrames, the first one for input
            columns, and the second for output columns.

        Examples
        --------
        >>> df_input, df_output = experiment_data.to_pandas()
        """
        if keep_references:
            return (
                pd.DataFrame([es._input_data for _, es in self],
                             index=self.index),
                pd.DataFrame([es._output_data for _, es in self],
                             index=self.index)
            )
        else:
            return (
                pd.DataFrame([es.input_data for _, es in self],
                             index=self.index),
                pd.DataFrame([es.output_data for _, es in self],
                             index=self.index)
            )

    def to_xarray(self, keep_references: bool = False) -> xr.Dataset:
        """
        Convert the ExperimentData object to an xarray Dataset.

        Parameters
        ----------
        keep_references : bool, optional
            If True, the references to the output data are kept, by default
            False.

        Returns
        -------
        xr.Dataset
            An xarray Dataset containing the data.

        Examples
        --------
        >>> dataset = experiment_data.to_xarray()
        """
        df_input, df_output = self.to_pandas(keep_references=keep_references)

        da_input = xr.DataArray(df_input, dims=['iterations', 'input_dim'],
                                coords={'iterations': self.index,
                                        'input_dim': df_input.columns})

        da_output = xr.DataArray(df_output, dims=['iterations', 'output_dim'],
                                 coords={'iterations': self.index,
                                         'output_dim': df_output.columns})

        return xr.Dataset({'input': da_input, 'output': da_output})

    # TODO: Implement this
    def get_n_best_output(self, n_samples: int,
                          output_name: Optional[str] = 'y') -> ExperimentData:
        """
        Get the n best samples from the output data. Lower values are better.

        Parameters
        ----------
        n_samples : int
            Number of samples to select.
        output_name : str, optional
            The name of the output column to sort by, by default 'y'.

        Returns
        -------
        ExperimentData
            New ExperimentData object with a selection of the n best samples.

        Examples
        --------
        >>> best_samples = experiment_data.get_n_best_output(5)
        """
        _, df_out = self.to_pandas()
        indices = df_out.nsmallest(n=n_samples, columns=output_name).index
        return self[indices]

    def to_multiindex(self) -> pd.DataFrame:
        """
        Convert the ExperimentData object to a pandas DataFrame with a
        MultiIndex. This is used for visualization purposes in a Jupyter
        notebook environment.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame with a MultiIndex.

        Examples
        --------
        >>> df_multiindex = experiment_data.to_multiindex()
        """
        list_of_dicts = [sample.to_multiindex() for _, sample in self]
        return pd.DataFrame(merge_dicts(list_of_dicts), index=self.index)

    #                                                     Append or remove data
    # =========================================================================

    def add_experiments(self,
                        data: ExperimentSample | ExperimentData
                        ) -> None:
        """
        Add an ExperimentSample or ExperimentData to the ExperimentData
        attribute.

        Parameters
        ----------
        data : ExperimentSample or ExperimentData
            Experiment(s) to add.

        Raises
        ------
        ValueError
            If the input is not an ExperimentSample or ExperimentData object.

        Examples
        --------
        >>> experiment_data.add_experiments(new_sample)
        >>> experiment_data.add_experiments(new_data)
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

    # Not used
    def overwrite(
        self, indices: Iterable[int],
            domain: Optional[Domain] = None,
            input_data: Optional[
                pd.DataFrame | np.ndarray
                | List[Dict[str, Any]] | str | Path] = None,
            output_data: Optional[
                pd.DataFrame | np.ndarray
                | List[Dict[str, Any]] | str | Path] = None,
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
        raise NotImplementedError()

    # Not used
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
        raise NotImplementedError()

    # Used in parallel mode
    def overwrite_disk(
        self, indices: Iterable[int],
            domain: Optional[Domain] = None,
            input_data: Optional[
                pd.DataFrame | np.ndarray
                | List[Dict[str, Any]] | str | Path] = None,
            output_data: Optional[
                pd.DataFrame | np.ndarray
                | List[Dict[str, Any]] | str | Path] = None,
            jobs: Optional[Path | str] = None,
            add_if_not_exist: bool = False
    ) -> None:
        raise NotImplementedError()

    def remove_rows_bottom(self, number_of_rows: int):
        """
        Remove a number of rows from the end of the ExperimentData object.

        Parameters
        ----------
        number_of_rows : int
            Number of rows to remove from the bottom.

        Examples
        --------
        >>> experiment_data.remove_rows_bottom(3)
        """
        # remove the last n rows
        for i in range(number_of_rows):
            self.data.pop(self.index[-1])

    def reset_index(self) -> ExperimentData:
        """
        Reset the index of the ExperimentData object.
        The index will be reset to a range from 0 to the number of experiments.

        Returns
        -------
        ExperimentData
            ExperimentData object with a reset index.

        Examples
        --------
        >>> reset_data = experiment_data.reset_index()
        """
        return ExperimentData.from_data(
            data={i: v for i, v in enumerate(self.data.values())},
            domain=self.domain,
            project_dir=self.project_dir)

    def join(self, experiment_data: ExperimentData) -> ExperimentData:
        """
        Join two ExperimentData objects.

        Parameters
        ----------
        experiment_data : ExperimentData
            The other ExperimentData object to join with.

        Returns
        -------
        ExperimentData
            The joined ExperimentData object.

        Examples
        --------
        >>> joined_data = experiment_data1.join(experiment_data2)
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

    def replace_nan(self, value: Any):
        """
        Replace all NaN values in the output data with the given value.

        Parameters
        ----------
        value : Any
            The value to replace NaNs with.

        Examples
        --------
        >>> experiment_data.replace_nan(0)
        """
        for _, es in self:
            es.replace_nan(value)

    def round(self, decimals: int):
        """
        Round all output data to the given number of decimals.

        Parameters
        ----------
        decimals : int
            Number of decimals to round to.

        Examples
        --------
        >>> experiment_data.round(2)
        """
        for _, es in self:
            es.round(decimals)

    #                                                          ExperimentSample
    # =========================================================================

    def get_experiment_sample(self, id: int) -> ExperimentSample:
        """
        Gets the experiment_sample at the given index.

        Parameters
        ----------
        id : int
            The index of the experiment_sample to retrieve.

        Returns
        -------
        ExperimentSample
            The ExperimentSample at the given index.

        Examples
        --------
        >>> sample = experiment_data.get_experiment_sample(0)
        """
        return self.data[id]

    def store_experimentsample(
            self, experiment_sample: ExperimentSample, id: int):
        """
        Store an ExperimentSample object in the ExperimentData object.

        Parameters
        ----------
        experiment_sample : ExperimentSample
            The ExperimentSample object to store.
        id : int
            The index of the ExperimentSample object.

        Examples
        --------
        >>> experiment_data.store_experimentsample(sample, 0)
        """
        self.domain += experiment_sample.domain

        for name, value in experiment_sample._output_data.items():

            # # If the output parameter is not in the domain, add it
            # if name not in self.domain.output_names:
            #     self.domain.add_output(name=name, to_disk=True)

            parameter = self.domain.output_space[name]

            # If the parameter is to be stored on disk, store it
            # Also check if the value is not already a reference!
            if parameter.to_disk and not isinstance(value, (Path, str)):
                storage_location = store_to_disk(
                    project_dir=self.project_dir, object=value, name=name,
                    id=id, store_function=parameter.store_function)

                experiment_sample._output_data[name] = Path(storage_location)

        for name, value in experiment_sample._input_data.items():

            # # If the output parameter is not in the domain, add it
            # if name not in self.domain.output_names:
            #     self.domain.add_output(name=name, to_disk=True)

            parameter = self.domain.input_space[name]

            # If the parameter is to be stored on disk, store it
            # Also check if the value is not already a reference!
            if parameter.to_disk and not isinstance(value, (Path, str)):
                storage_location = store_to_disk(
                    project_dir=self.project_dir, object=value, name=name,
                    id=id, store_function=parameter.store_function)

                experiment_sample._input_data[name] = Path(storage_location)

        # Set the experiment sample in the ExperimentData object
        self.data[id] = experiment_sample

    # Used in parallel mode

    def get_open_job(self) -> Tuple[int, ExperimentSample]:
        """
        Get the first open job in the ExperimentData object.

        Returns
        -------
        tuple of int and ExperimentSample
            The index and ExperimentSample of the first open job.

        Notes
        -----
        This function iterates over the ExperimentData object and returns the
        first open job. If no open jobs are found, it returns None.

        The returned open job is marked as 'in_progress'.

        Examples
        --------
        >>> job_id, job_sample = experiment_data.get_open_job()
        """
        for id, es in self:
            if es.is_status('open'):
                es.mark('in_progress')
                return id, es

        return None, ExperimentSample()

    #                                                                      Jobs
    # =========================================================================

    def is_all_finished(self) -> bool:
        """
        Check if all jobs are finished.

        Returns
        -------
        bool
            True if all jobs are finished, False otherwise.

        Examples
        --------
        >>> experiment_data.is_all_finished()
        True
        """
        return all(es.is_status('finished') for _, es in self)

    def mark(self, indices: int | Iterable[int],
             status: Literal['open', 'in_progress', 'finished', 'error']):
        """
        Mark the jobs at the given indices with the given status.

        Parameters
        ----------
        indices : int or Iterable[int]
            Indices of the jobs to mark.
        status : {'open', 'in_progress', 'finished', 'error'}
            Status to mark the jobs with.

        Raises
        ------
        ValueError
            If the given status is not valid.

        Examples
        --------
        >>> experiment_data.mark([0, 1], 'finished')
        """
        if isinstance(indices, int):
            indices = [indices]
        for i in indices:
            self.data[i].mark(status)

    def mark_all(self,
                 status: Literal['open', 'in_progress', 'finished', 'error']):
        """
        Mark all the experiments with the given status.

        Parameters
        ----------
        status : {'open', 'in_progress', 'finished', 'error'}
            Status to mark the jobs with.

        Raises
        ------
        ValueError
            If the given status is not valid.

        Examples
        --------
        >>> experiment_data.mark_all('finished')
        """
        for _, es in self:
            es.mark(status)

    def run(self, block: Block | Iterable[Block], **kwargs) -> ExperimentData:
        """
        Run a block over the entire ExperimentData object.

        Parameters
        ----------
        block : Block
            The block(s) to run.
        **kwargs
            Additional keyword arguments passed to the block.

        Returns
        -------
        ExperimentData
            The ExperimentData object after running the block.

        Examples
        --------
        >>> experiment_data.run(block)
        """
        if isinstance(block, Block):
            block = [block]

        for b in block:
            b.arm(data=self)
            self = b.call(**kwargs)

        return self

    #                                                            Datageneration
    # =========================================================================

    def evaluate(self, data_generator: Block | str,
                 mode: Literal['sequential', 'parallel',
                               'cluster', 'cluster_parallel'] = 'sequential',
                 output_names: Optional[List[str]] = None,
                 **kwargs) -> None:
        """
        Run any function over the entirety of the experiments.

        Parameters
        ----------
        data_generator : DataGenerator
            Data generator to use.
        mode : {'sequential', 'parallel', 'cluster', 'cluster_parallel'},
          optional
            Operational mode, by default 'sequential'.
        output_names : list of str, optional
            Names of the output parameters, by default None.
        **kwargs
            Additional keyword arguments passed to the data generator.

        Raises
        ------
        ValueError
            If an invalid parallelization mode is specified.

        Examples
        --------
        >>> experiment_data.evaluate(data_generator, mode='parallel')
        """
        # Create
        data_generator = _datagenerator_factory(
            data_generator=data_generator, output_names=output_names, **kwargs)

        self = self.run(block=data_generator, mode=mode, **kwargs)

    #                                                              Optimization
    # =========================================================================

    def optimize(self, optimizer: Block | str,
                 data_generator: DataGenerator | str,
                 iterations: int,
                 kwargs: Optional[Dict[str, Any]] = None,
                 hyperparameters: Optional[Dict[str, Any]] = None,
                 x0_selection: Literal['best', 'random',
                                       'last',
                                       'new'] | ExperimentData = 'best',
                 sampler: Optional[Block | str] = 'random',
                 overwrite: bool = False,
                 callback: Optional[Callable] = None) -> None:
        """
        Optimize the ExperimentData object.

        Parameters
        ----------
        optimizer : Block or str or Callable
            Optimizer object.
        data_generator : DataGenerator or str
            DataGenerator object.
        iterations : int
            Number of iterations.
        kwargs : dict, optional
            Additional keyword arguments passed to the DataGenerator.
        hyperparameters : dict, optional
            Additional keyword arguments passed to the optimizer.
        x0_selection : {'best', 'random', 'last', 'new'} or ExperimentData
            How to select the initial design, by default 'best'.
        sampler : Block or str, optional
            Sampler to use if x0_selection is 'new', by default 'random'.
        overwrite : bool, optional
            If True, the optimizer will overwrite the current data, by default
            False.
        callback : Callable, optional
            A callback function called after every iteration.

        Raises
        ------
        ValueError
            If an invalid x0_selection is specified.

        Examples
        --------
        >>> experiment_data.optimize(optimizer, data_generator, iterations=10)
        """
        if kwargs is None:
            kwargs = {}

        # Create the data generator object if a string reference is passed
        if isinstance(data_generator, str):
            data_generator = _datagenerator_factory(
                data_generator=data_generator, **kwargs)

        if hyperparameters is None:
            hyperparameters = {}

        # Create the optimizer object if a string reference is passed
        if isinstance(optimizer, str):
            optimizer = _optimizer_factory(
                optimizer=optimizer, **hyperparameters)

        # Create the sampler object if a string reference is passed
        if isinstance(sampler, str):
            sampler = _sampler_factory(sampler=sampler)

        last_index = self.index[-1] if not self.index.empty else -1

        population = optimizer.population if hasattr(
            optimizer, 'population') else 1
        seed = optimizer.seed if hasattr(optimizer, 'seed') else None
        opt_type = optimizer.type if hasattr(optimizer, 'type') else None

        if isinstance(x0_selection, str):
            if x0_selection == 'new':

                if iterations < population:
                    raise ValueError(
                        f'For creating new samples, the total number of '
                        f'requested iterations ({iterations}) cannot be '
                        f'smaller than the population size '
                        f'({population})')

                init_samples = ExperimentData(
                    domain=self.domain,
                    project_dir=self.project_dir)

                init_samples.sample(sampler=sampler,
                                    n_samples=population,
                                    seed=seed)

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
                iterations -= population

        x0 = x0_factory(experiment_data=self, mode=x0_selection,
                        n_samples=population)

        x0.evaluate(data_generator=data_generator, mode='sequential',
                    **kwargs)

        if len(x0) < population:
            raise ValueError((
                f"There are {len(self.data)} samples available, "
                f"need {population} for initial population!"
            ))

        optimizer.arm(data=x0)
        n_updates = range(
            iterations // population + (iterations % population > 0))

        if opt_type == 'scipy':
            # Scipy optimizers work differently since they are not
            # able to output a single update step
            optimizer._iterate(data_generator=data_generator,
                               iterations=iterations, kwargs=kwargs,
                               overwrite=overwrite)
        else:
            for _ in n_updates:
                optimizer.data += optimizer.call(grad_fn=data_generator.dfdx)
                optimizer.data.evaluate(data_generator=data_generator,
                                        mode='sequential',
                                        **kwargs)

        optimizer.data.remove_rows_bottom(
            number_of_rows=population * n_updates.stop - iterations)

        self._add(experiment_data=optimizer.data[population:])

    #                                                                  Sampling
    # =========================================================================

    def sample(self, sampler: Block | str, **kwargs) -> None:
        """
        Sample data from the domain providing the sampler strategy

        Parameters
        ----------
        sampler: BlockAbstract | str
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

        samples = self.run(block=sampler, **kwargs)

        self._add(samples)

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

        Examples
        --------
        >>> experiment_data.remove_lockfile()
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
                              jobs=subdirectory / JOBS_FILENAME)

    except FileNotFoundError:
        raise FileNotFoundError(
            f"Cannot find the files from {subdirectory}.")


def convert_numpy_to_dataframe_with_domain(
        array: np.ndarray, names: Optional[List[str]], mode: str
) -> pd.DataFrame:
    if not names:
        if mode == 'input':
            names = [f'x{i}' for i in range(array.shape[1])]
        elif mode == 'output':
            if array.shape[1] == 1:
                names = ['y']
            else:
                names = [f'y{i}' for i in range(array.shape[1])]

        else:
            raise ValueError(
                f"Unknown mode {mode}, use 'input' or 'output'")

    return pd.DataFrame(array, columns=names)


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


def _dict_factory(data: pd.DataFrame | List[Dict[str, Any]] | None | Path | str
                  ) -> List[Dict[str, Any]]:
    """
    Convert the DataTypes to a list of dictionaries

    Parameters
    ----------
    data : pd.DataFrame | List[Dict[str, Any]] | None | Path | str
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

    elif isinstance(data, (Path, str)):
        return _dict_factory(pd.read_csv(
            Path(data).with_suffix('.csv'),
            header=0, index_col=0))

    # check if data is already a list of dicts
    elif isinstance(data, list) and all(isinstance(d, dict) for d in data):
        return data

    # If the data is a pandas DataFrame, convert it to a list of dictionaries
    elif isinstance(data, pd.DataFrame):
        return [row._asdict() for row in data.itertuples(index=False)]

    raise ValueError(f"Data type {type(data)} not supported")


def data_factory(input_data: List[Dict[str, Any]],
                 output_data: List[Dict[str, Any]],
                 domain: Domain,
                 jobs: pd.Series,
                 project_dir: Path,
                 ) -> Dict[int, ExperimentSample]:
    """
    Convert the input and output data to a defaultdictionary
    of ExperimentSamples

    Parameters
    ----------
    input_data : List[Dict[str, Any]]
        The input data of the experiments
    output_data : List[Dict[str, Any]]
        The output data of the experiments
    domain : Domain
        The domain of the data
    jobs : pd.Series
        The status of all the jobs
    project_dir : Path
        The project directory of the data


    Returns
    -------
    Dict[int, ExperimentSample]
        The converted data as a defaultdict of ExperimentSamples

    """
    # Combine the two lists into a dictionary of ExperimentSamples
    data = {index: ExperimentSample(input_data=experiment_input,
                                    output_data=experiment_output,
                                    domain=domain,
                                    job_status=job_status,
                                    project_dir=project_dir)
            for index, (experiment_input, experiment_output, job_status) in
            enumerate(zip_longest(input_data, output_data, jobs))}

    return defaultdict(ExperimentSample, data)


def jobs_factory(jobs: pd.Series | str | Path | None) -> pd.Series:
    if isinstance(jobs, pd.Series):
        return jobs

    elif jobs is None:
        return pd.Series()

    elif isinstance(jobs, (Path, str)):
        return pd.read_csv(
            Path(jobs).with_suffix('.csv'),
            header=0, index_col=0).squeeze()

    else:
        raise ValueError(f"Jobs type {type(jobs)} not supported")
