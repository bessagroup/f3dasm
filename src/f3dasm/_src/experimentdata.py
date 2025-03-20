"""
The ExperimentData object is the main object used to store implementations
 of a design-of-experiments, keep track of results, perform optimization and
 extract data for machine learning purposes.
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
import random
from collections import defaultdict
from copy import copy
from itertools import zip_longest
from pathlib import Path
from time import sleep
from typing import (Any, Callable, Dict, Iterable, Iterator, List, Literal,
                    Optional, Protocol, Tuple, Type)

# Third-party
import numpy as np
import pandas as pd
import xarray as xr
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

# Local
from ._io import (DOMAIN_FILENAME, EXPERIMENTDATA_SUBFOLDER,
                  INPUT_DATA_FILENAME, JOBS_FILENAME, MAX_TRIES,
                  OUTPUT_DATA_FILENAME, _project_dir_factory)
from .design.domain import Domain, _domain_factory
from .errors import DecodeError, EmptyFileError, ReachMaximumTriesError
from .experimentsample import ExperimentSample
from .logger import logger

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

#                                                                      Protocol
# =============================================================================


class Block(Protocol):
    def arm(self, data: ExperimentData) -> None:
        ...

    def call(self, data: ExperimentData, **kwargs) -> ExperimentData:
        ...


class DataGenerator(Block):
    ...

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
        Main object to store implementations of a design-of-experiments, keep
        track of results, perform optimization and extract data for machine
        learning purposes.

        Parameters
        ----------
        domain : Domain, optional
            The domain of the experiment, by default None.
        input_data : pd.DataFrame | np.ndarray | List[Dict[str, Any]] |
                    str | Path, optional
            The input data of the experiment, by default None.
        output_data : pd.DataFrame | np.ndarray | List[Dict[str, Any]] |
                     str | Path, optional
            The output data of the experiment, by default None.
        jobs : pd.Series, optional
            The status of all the jobs, by default None.
        project_dir : Path, optional
            Directory of the project, by default None.

        Examples
        --------
        >>> experiment_data = ExperimentData(
        ...     domain=domain_obj,
        ...     input_data=input_df,
        ...     output_data=output_df
        ... )
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
        self._domain = _domain
        self.project_dir = _project_dir

        # Store to_disk objects so that the references are kept only
        for id, experiment_sample in self:
            experiment_sample.store_experimentsample_references(idx=id)

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
        return (self.data == __o.data and self._domain == __o._domain
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
            domain=self._domain,
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

    def __deepcopy__(self) -> ExperimentData:
        """
        Returns a deep copy of the ExperimentData object.

        Returns
        -------
        ExperimentData
            Deep copy of the ExperimentData object.

        Examples
        --------
        >>> from copy import deepcopy
        >>> copied_data = deepcopy(experiment_data)
        """
        return self._copy(in_place=False, deep=True)

    def __copy__(self) -> ExperimentData:
        """
        Returns a shallow copy of the ExperimentData object.

        Returns
        -------
        ExperimentData
            Shallow copy of the ExperimentData object.

        Examples
        --------
        >>> from copy import copy
        >>> copied_data = copy(experiment_data)
        """
        return self._copy(in_place=False, deep=False)
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

    @property
    def domain(self) -> Domain:
        return self._domain

    @domain.setter
    def domain(self, domain: Domain):
        """
        Sets the domain of the ExperimentData object.

        Parameters
        ----------
        domain : Domain
            The domain to set.
        """
        self._domain = domain
        for _, es in self:
            es.domain = domain

    #                                                  Alternative constructors
    # =========================================================================

    @classmethod
    def _from_attributes(cls: Type[ExperimentData],
                         domain: Domain,
                         data: Dict[int, ExperimentSample],
                         project_dir: Path) -> ExperimentData:
        """
        Create an ExperimentData object from attributes.

        Parameters
        ----------
        domain : Domain
            The domain of the data.
        data : dict of int to ExperimentSample
            The data of the experiment.
        project_dir : Path
            The project directory.

        Returns
        -------
        ExperimentData
            ExperimentData object containing the loaded data.
        """
        experiment_data = cls()
        experiment_data.data = data
        experiment_data._domain = domain
        experiment_data.project_dir = project_dir
        return experiment_data

    @classmethod
    def from_file(cls: Type[ExperimentData],
                  project_dir: Path | str,
                  wait_for_creation: bool = False,
                  max_tries: int = MAX_TRIES) -> ExperimentData:
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
            return _from_file_attempt(project_dir=project_dir,
                                      wait_for_creation=wait_for_creation,
                                      max_tries=max_tries)
        except FileNotFoundError:
            try:
                filename_with_path = Path(get_original_cwd()) / project_dir
            except ValueError:  # get_original_cwd() hydra initialization error
                raise FileNotFoundError(
                    f"Cannot find the folder {project_dir} !")

            return _from_file_attempt(project_dir=filename_with_path,
                                      wait_for_creation=wait_for_creation,
                                      max_tries=max_tries)

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
        # Option 1: From existing ExperimentData files
        if 'from_file' in config:
            return cls.from_file(config.from_file)

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
        experiment_data._domain = domain
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

    def _copy(self, in_place: bool = False,
              deep: bool = True) -> ExperimentData:
        """
        Create a copy of the ExperimentData object.

        Parameters
        ----------
        in_place : bool, optional
            If True, no copy is made and the object itself is returned,
            by default False.
        deep : bool, optional
            If True, a deep copy is made, by default True

        Returns
        -------
        ExperimentData
            A copy of the ExperimentData object or the original object

        Examples
        --------
        >>> copied_data = experiment_data._copy(in_place=False)
        """
        if in_place:
            return self

        if deep:
            data_copy = {k: v._copy() for k, v in self.data.items()}
        else:
            data_copy = self.data

        return ExperimentData._from_attributes(
            data=defaultdict(ExperimentSample, data_copy),
            domain=self._domain._copy(),
            project_dir=self.project_dir,
        )

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
        * the domain (`domain.json`)

        To avoid the ExperimentData to be written simultaneously by multiple
        processes, a '.lock' file is automatically created
        in the project directory. Concurrent process can only sequentially
        access the lock file. This lock file is removed after the
        ExperimentData object is written to disk.
        """
        if project_dir is not None:
            self.set_project_dir(project_dir, in_place=True)

        subdirectory = self.project_dir / EXPERIMENTDATA_SUBFOLDER

        # Create the experimentdata subfolder if it does not exist
        subdirectory.mkdir(parents=True, exist_ok=True)

        df_input, df_output = self.to_pandas(keep_references=True)

        df_input.to_csv(
            (subdirectory / INPUT_DATA_FILENAME).with_suffix('.csv'))
        df_output.to_csv(
            (subdirectory / OUTPUT_DATA_FILENAME).with_suffix('.csv'))
        self._domain.store(subdirectory / DOMAIN_FILENAME)
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
                        data: ExperimentSample | ExperimentData,
                        in_place: bool = False,
                        ) -> None:
        """
        Add an ExperimentSample or ExperimentData to the ExperimentData
        attribute.

        Parameters
        ----------
        data : ExperimentSample or ExperimentData
            Experiment(s) to add.
        in_place : bool, optional
            If True, the data is added in place, by default False.

        Raises
        ------
        ValueError
            If the input is not an ExperimentSample or ExperimentData object.

        Examples
        --------
        >>> experiment_data.add_experiments(new_sample)
        >>> experiment_data.add_experiments(new_data)
        """
        d = self._copy(in_place=in_place)

        if isinstance(data, ExperimentSample):
            d._add_experiment_sample(data)

        elif isinstance(data, ExperimentData):
            d._add(data)

        else:
            raise ValueError((
                f"The input to this function should be an ExperimentSample or "
                f"ExperimentData object, not {type(data)} ")
            )

        if in_place:
            return None
        else:
            return d

    def remove_rows_bottom(self, number_of_rows: int, in_place: bool = False):
        """
        Remove a number of rows from the end of the ExperimentData object.

        Parameters
        ----------
        number_of_rows : int
            Number of rows to remove from the bottom.
        in_place : bool, optional
            If True, the rows are removed in place, by default False.

        Examples
        --------
        >>> experiment_data.remove_rows_bottom(3)
        """
        d = self._copy(in_place=in_place)

        # remove the last n rows
        for i in range(number_of_rows):
            d.data.pop(self.index[-1])

        if in_place:
            return None
        else:
            return d

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
            domain=self._domain,
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

        copy_self._domain += copy_other._domain

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
        self._domain += copy_other._domain

    def _add_experiment_sample(self, experiment_sample: ExperimentSample):
        last_key = max(self.index) if self else -1
        self.data[last_key + 1] = experiment_sample

    def replace_nan(self, value: Any, in_place: bool = False):
        """
        Replace all NaN values in the output data with the given value.

        Parameters
        ----------
        value : Any
            The value to replace NaNs with.
        in_place : bool, optional
            If True, the NaN values are replaced in place, by default False.

        Examples
        --------
        >>> experiment_data.replace_nan(0)
        """
        d = self._copy(in_place=in_place)
        for _, es in d:
            es.replace_nan(value)

        if in_place:
            return None
        else:
            return d

    def round(self, decimals: int, in_place: bool = False):
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
        d = self._copy(in_place=in_place)

        for _, es in self:
            es.round(decimals)

        if in_place:
            return None
        else:
            return d

    # TODO: Create tests for this
    def sort(self, criterion: Callable[[ExperimentSample], Any],
             reverse: bool = False) -> ExperimentData:
        """
        Sort the ExperimentData object based on a criterion.

        Parameters
        ----------
        criterion : Callable[[ExperimentSample], Any]
            The criterion to sort on. This should be a function that takes an
            ExperimentSample object and returns a value to sort on.
        reverse : bool, optional
            If True, sort in descending order, by default False.

        Returns
        -------
        ExperimentData
            The sorted ExperimentData object.

        Examples
        --------
        >>> sorted_data = experiment_data.sort(lambda x: x.output_data['y'])
        """

        sorted_data = dict(
            sorted(self.data.items(), key=lambda item: criterion(
                item[1]), reverse=reverse)
        )
        return ExperimentData.from_data(
            data=sorted_data,
            domain=self._domain,
            project_dir=self.project_dir
        )

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
            self, experiment_sample: ExperimentSample, idx: int):
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
        self._domain += experiment_sample.domain
        self.data[idx] = experiment_sample

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
             status: Literal['open', 'in_progress', 'finished', 'error'],
             in_place: bool = False):
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
        d = self._copy(in_place=in_place)

        if isinstance(indices, int):
            indices = [indices]
        for i in indices:
            d.data[i].mark(status)

        if in_place:
            return None
        else:
            return d

    def mark_all(self,
                 status: Literal['open', 'in_progress', 'finished', 'error'],
                 in_place: bool = False):
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
        d = self._copy(in_place=in_place)

        for _, es in d:
            es.mark(status)

        if in_place:
            return None
        else:
            return d

    #                                                              Optimization
    # =========================================================================

    def optimize(self, optimizer: Block,
                 data_generator: DataGenerator,
                 iterations: int,
                 x0_selection: Literal['best', 'random',
                                       'last',
                                       'new'] | ExperimentData = 'best',
                 kwargs: Optional[Dict[str, Any]] = None,
                 sampler: Optional[Block] = None,
                 overwrite: bool = False) -> None:
        """
        Optimize the ExperimentData object.

        Parameters
        ----------
        optimizer : Block
            Optimizer object.
        data_generator : DataGenerator
            DataGenerator object.
        iterations : int
            Number of iterations.
        x0_selection : {'best', 'random', 'last', 'new'} or ExperimentData
            How to select the initial design, by default 'best'.
        sampler : Block, optional
            Sampler to use if x0_selection is 'new', by default None.
        overwrite : bool, optional
            If True, the optimizer will overwrite the current data, by default
            False.

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

        population = optimizer.population if hasattr(
            optimizer, 'population') else 1

        opt_type = optimizer.type if hasattr(optimizer, 'type') else None

        if iterations < population and x0_selection == 'new':
            raise ValueError(
                f'For creating new samples, the total number of '
                f'requested iterations ({iterations}) cannot be '
                f'smaller than the population size '
                f'({population})')

        x0 = x0_factory(experiment_data=self, mode=x0_selection,
                        n_samples=population, sampler=sampler)

        x0 = data_generator.call(data=x0, mode='sequential')

        if len(x0) < population:
            raise ValueError((
                f"There are {len(self.data)} samples available, "
                f"need {population} for initial population!"
            ))

        optimizer.arm(data=x0)
        data_generator.arm(data=x0)

        x = x0

        n_updates = range(
            iterations // population + (iterations % population > 0))

        if opt_type == 'scipy':
            # Scipy optimizers work differently since they are not
            # able to output a single update step
            optimizer._iterate(data=x, data_generator=data_generator,
                               iterations=iterations, kwargs=kwargs,
                               overwrite=overwrite)
        else:
            for _ in n_updates:
                if overwrite:
                    x = optimizer.call(data=x,
                                       grad_fn=data_generator.dfdx)
                else:
                    x += optimizer.call(data=x,
                                        grad_fn=data_generator.dfdx)
                x = data_generator.call(data=x, mode='sequential',
                                        **kwargs)

        if not overwrite:
            x.remove_rows_bottom(
                number_of_rows=population * n_updates.stop - iterations,
                in_place=True)
            self._add(experiment_data=x[population:])

        else:
            self._add(experiment_data=x)

    #                                                         Project directory
    # =========================================================================

    def set_project_dir(self, project_dir: Path | str,
                        in_place: bool = False) -> ExperimentData:
        """Set the directory of the f3dasm project folder.

        Parameters
        ----------
        project_dir : Path or str
            Path to the project directory
        in_place : bool, optional
            If True, the project directory is set in place, by default False

        Returns
        -------
        ExperimentData
            ExperimentData object with the updated project directory
        """
        d = self._copy(in_place=in_place)
        d.project_dir = _project_dir_factory(project_dir)

        if in_place:
            return None
        else:
            return d

    def move_project_dir(self, project_dir: Path | str):

        Path(project_dir).mkdir(parents=True, exist_ok=True)

        for _, es in self:
            es.copy_project_dir(Path(project_dir))

        self.set_project_dir(project_dir, in_place=True)

# =============================================================================


def x0_factory(experiment_data: ExperimentData,
               mode: str | ExperimentData, n_samples: int,
               sampler: Block) -> ExperimentData:
    """Set the initial population to the best n samples of the given data

    Parameters
    ----------
    experiment_data : ExperimentData
        Data to be used for the initial population
    mode : str
        Mode of selecting the initial population, by default 'best'
        The following modes are available:

            - best: select the best n samples
            - new: create n new samples
            - random: select n random samples
            - last: select the last n samples
    n_samples : int
        Number of samples to select
    sampler : Block or str
        Sampler object containing the sampling strategy or one of the
        built-in sampler names. Used if mode is 'new'

    Returns
    -------
    ExperimentData
        Initial population of the optimization

    Raises
    ------
    ValueError
        Raises when the mode is not recognized
    """
    if isinstance(mode, ExperimentData):
        x0 = mode

    if mode == 'new':
        sampler.arm(data=experiment_data)
        x0 = sampler.call(data=experiment_data, n_samples=n_samples)

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


def _from_file_attempt(project_dir: Path, max_tries: int = MAX_TRIES,
                       wait_for_creation: bool = False
                       ) -> ExperimentData:
    """Attempt to create an ExperimentData object
    from .csv and .pkl files.

    Parameters
    ----------
    project_dir : Path
        Name of the user-defined directory where the files are stored.
    max_tries : int, optional
        Maximum number of tries to read the files, by default MAX_TRIES
    wait_for_creation : bool, optional
        If True, wait for the files to be created, by default False

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

    # Retrieve the updated experimentdata object from disc
    tries = 0
    while tries <= max_tries:
        try:
            return ExperimentData(
                domain=subdirectory / DOMAIN_FILENAME,
                input_data=subdirectory / INPUT_DATA_FILENAME,
                output_data=subdirectory / OUTPUT_DATA_FILENAME,
                jobs=subdirectory / JOBS_FILENAME,
                project_dir=project_dir)
        except (EmptyFileError, DecodeError):
            tries += 1
            logger.debug((
                f"Error reading a file, retrying"
                f" {tries+1}/{MAX_TRIES}"
            ))
            sleep(random.uniform(0.5, 2.5))

        except FileNotFoundError:
            if not wait_for_creation:
                raise FileNotFoundError(
                    f"File {subdirectory} not found")

            tries += 1
            logger.debug((
                f"FileNotFoundError({subdirectory}), sleeping!"
            ))
            sleep(random.uniform(9.5, 11.0))

    raise ReachMaximumTriesError(file_path=subdirectory, max_tries=max_tries)


def convert_numpy_to_dataframe_with_domain(
        array: np.ndarray, names: Optional[List[str]],
        mode: Literal['input', 'output']
) -> pd.DataFrame:
    """
    Convert a numpy array to a pandas DataFrame with the domain names

    Parameters
    ----------
    array : np.ndarray
        The numpy array to be converted
    names : List[str], optional
        The names of the columns, by default None
    mode : str
        The mode of the data, either 'input' or 'output'

    Returns
    -------
    pd.DataFrame
        The converted data as a pandas DataFrame
    """
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
        filepath = Path(data).with_suffix('.csv')

        if not filepath.exists():
            raise FileNotFoundError(f"File {filepath} not found")

        if filepath.stat().st_size == 0:
            raise EmptyFileError(filepath)

        try:
            df = pd.read_csv(filepath, header=0, index_col=0)
        except pd.errors.EmptyDataError:
            raise DecodeError(filepath)

        return _dict_factory(df)

    # check if data is already a list of dicts
    elif isinstance(data, list) and all(isinstance(d, dict) for d in data):
        return data

    # If the data is a pandas DataFrame, convert it to a list of dictionaries
    # Note : itertuples() is faster but renames the column names
    elif isinstance(data, pd.DataFrame):
        return [row.to_dict() for _, row in data.iterrows()]

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
    # remove all key-value pairs that have a None or np.nan value
    input_data = remove_nan_and_none_keys_inplace(input_data)
    output_data = remove_nan_and_none_keys_inplace(output_data)
    # Combine the two lists into a dictionary of ExperimentSamples
    data = {index: ExperimentSample(input_data=experiment_input,
                                    output_data=experiment_output,
                                    domain=domain,
                                    job_status=job_status,
                                    project_dir=project_dir)
            for index, (experiment_input, experiment_output, job_status) in
            enumerate(zip_longest(input_data, output_data, jobs))}

    return defaultdict(ExperimentSample, data)


def remove_nan_and_none_keys_inplace(data_list: List[Dict[str, Any]]) -> None:
    for data in data_list:
        keys_to_remove = [k for k, v in data.items() if v is None or (
            isinstance(v, float) and np.isnan(v))]
        for k in keys_to_remove:
            del data[k]

    return data_list


def jobs_factory(jobs: pd.Series | str | Path | None) -> pd.Series:
    if isinstance(jobs, pd.Series):
        return jobs

    elif jobs is None:
        return pd.Series()

    elif isinstance(jobs, (Path, str)):

        filepath = Path(jobs).with_suffix('.csv')

        if not filepath.exists():
            raise FileNotFoundError(f"File {filepath} not found")

        if filepath.stat().st_size == 0:
            raise EmptyFileError(filepath)

        try:
            df = pd.read_csv(filepath,
                             header=0, index_col=0).squeeze()
        except pd.errors.EmptyDataError:
            raise DecodeError(filepath)

        # If the jobs is jut one value, it is parsed as a string
        # So, make sure that we return a pd.Series either way!
        if not isinstance(df, pd.Series):
            df = pd.Series(df)

        return df

    else:
        raise ValueError(f"Jobs type {type(jobs)} not supported")
