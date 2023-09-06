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

if sys.version_info < (3, 8):
    from typing_extensions import Protocol
else:
    from typing import Protocol

from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Type

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
from ._jobqueue import FINISHED, OPEN, NoOpenJobsError, _JobQueue
from .design import Design
from .domain import Domain
from .parameter import Parameter

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class _Sampler(Protocol):
    """Protocol class for sampling methods."""
    def get_samples(numsamples: int) -> ExperimentData:
        ...

    @classmethod
    def from_yaml(cls, domain_config: DictConfig, sampler_config: DictConfig) -> '_Sampler':
        """Create a sampler from a yaml configuration"""

        args = {**sampler_config, 'design': None}
        sampler: _Sampler = instantiate(args)
        sampler.design = Domain.from_yaml(domain_config)
        return sampler


class _DesignCallable(Protocol):
    def __call__(design: Design, **kwargs) -> Design:
        ...


class ExperimentData:
    """
    A class that contains data for experiments.
    """

    def __init__(self, domain: Domain, filename: Optional[str] = 'experimentdata'):
        """
        Initializes an instance of ExperimentData.

        Parameters
        ----------
        domain : Domain
            A Domain object defining the input and output spaces of the experiment.
        filename : str, optional
            Name of the file, excluding suffix, by default 'experimentdata'.
        """
        self.domain = domain
        self.filename = filename
        self.input_data = _Data.from_domain(self.domain)
        self.output_data = _Data()
        self.jobs = _JobQueue.from_data(self.input_data)

    def __len__(self):
        """The len() method returns the number of datapoints"""
        return len(self.input_data)

    def __iter__(self) -> Iterator[Tuple[Dict[str, Any]]]:
        return self.input_data.__iter__()

    def __next__(self):
        return self.input_data.__next__()

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
    def from_file(cls: Type[ExperimentData], filename: str = 'experimentdata',
                  text_io: Optional[TextIOWrapper] = None) -> ExperimentData:
        """Create an ExperimentData object from .csv and .json files.

        Parameters
        ----------
        filename : str, optional
            Name of the file, excluding suffix, by default 'experimentdata'.
        text_io : TextIOWrapper or None, optional
            Text I/O wrapper object for reading the file, by default None.

        Returns
        -------
        ExperimentData
            ExperimentData object containing the loaded data.
        """
        try:
            return cls._from_file_attempt(filename, text_io)
        except FileNotFoundError:
            try:
                filename_with_path = Path(get_original_cwd()) / filename
            except ValueError:  # get_original_cwd() hydra initialization error
                raise FileNotFoundError(f"Cannot find the file {filename} !")

            return cls._from_file_attempt(filename_with_path, text_io)

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
    def from_dataframe(cls, dataframe_input: pd.DataFrame, dataframe_output: Optional[pd.DataFrame] = None,
                       domain: Optional[Domain] = None, filename: Optional[str] = 'experimentdata') -> ExperimentData:
        """Create an ExperimentData object from a pandas dataframe.

        Parameters
        ----------
        dataframe_input : pd.DataFrame
            Pandas dataframe containing the data with columns corresponding to the
            input parameter names
        dataframe_output : pd.DataFrame, optional
            Pandas dataframe containing the data with columns corresponding to the
            output parameter names, by default None
        domain : Domain, optional
            Domain object defining the input and output spaces of the experiment. If not given,
            the domain is inferred from the input data. By default None.
        filename : str, optional
            Name of the created experimentdata, excluding suffix, by default 'experimentdata'.

        Returns
        -------
        ExperimentData
            ExperimentData object containing the loaded data.
        """
        if domain is None:
            # Infer the domain from the input data
            domain = Domain.from_dataframe(dataframe_input)

        experimentdata = cls(domain=domain, filename=filename)
        experimentdata.input_data = _Data.from_dataframe(dataframe_input)

        if dataframe_output is not None:
            experimentdata.output_data = _Data.from_dataframe(dataframe_output)
        elif dataframe_output is None:
            experimentdata.output_data = _Data.from_indices(experimentdata.input_data.indices)

        experimentdata.jobs = _JobQueue.from_data(experimentdata.input_data)

        return experimentdata

    @classmethod
    def from_csv(cls, filename_input: Path, filename_output: Optional[Path] = None,
                 domain: Optional[Domain] = None) -> ExperimentData:
        """Create an ExperimentData object from .csv files.

        Parameters
        ----------
        filename_input : Path
            filename of the input .csv file.
        filename_output : Path, optional
            filename of the output .csv file, by default None
        domain : Domain, optional
            Domain object, by default None

        Returns
        -------
        ExperimentData
            ExperimentData object containing the loaded data.
        """
        # Read the input datat csv file as a pandas dataframe
        df_input = pd.read_csv(filename_input.with_suffix('.csv'), index_col=0)

        # Read the output data csv file as a pandas dataframe
        if filename_output is not None:
            df_output = pd.read_csv(filename_output.with_suffix('.csv'), index_col=0)
        else:
            df_output = None

        return cls.from_dataframe(df_input, df_output, domain, filename_input.stem)

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

        # Option 3: Import the csv file
        elif 'from_csv' in config.experimentdata:
            if 'domain' in config:
                domain = Domain.from_yaml(config.domain)
            else:
                domain = None

            return cls.from_csv(filename_input=config.experimentdata.from_csv.input_filepath,
                                filename_output=config.experimentdata.from_csv.output_filepath, domain=domain)

        else:
            raise ValueError("No valid experimentdata option found in the config file!")

    @classmethod
    def _from_file_attempt(cls: Type[ExperimentData], filename: str,
                           text_io: Optional[TextIOWrapper]) -> ExperimentData:
        """Attempt to create an ExperimentData object from .csv and .json files.

        Parameters
        ----------
        filename : str
            Name of the file, excluding suffix.
        text_io : TextIOWrapper or None
            Text I/O wrapper object for reading the file.

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
            domain = Domain.from_file(Path(f"{filename}_domain"))
            experimentdata = cls(domain=domain, filename=filename)
            experimentdata.input_data = _Data.from_file(Path(f"{filename}_data"), text_io)

            try:
                experimentdata.output_data = _Data.from_file(Path(f"{filename}_output"))
            except FileNotFoundError:
                experimentdata.output_data = _Data.from_indices(experimentdata.input_data.indices)

            experimentdata.jobs = _JobQueue.from_file(Path(f"{filename}_jobs"))
            return experimentdata
        except FileNotFoundError:
            raise FileNotFoundError(f"Cannot find the file {filename}_data.csv.")

    def reset_data(self):
        """Reset the dataframe to an empty dataframe with the appropriate input and output columns"""
        self.input_data.reset(self.domain)
        self.output_data.reset()
        self.jobs.reset()

    #                                                                        Export
    # =============================================================================

    def select(self, indices: List[int]) -> ExperimentData:
        """Select a subset of the data.

        Parameters
        ----------
        indices : List[int]
            List of indices to select.

        Returns
        -------
        ExperimentData
            ExperimentData object containing the selected data.
        """
        new_experimentdata = deepcopy(self)
        new_experimentdata.input_data.select(indices)
        new_experimentdata.output_data.select(indices)
        new_experimentdata.jobs.select(indices)

        return new_experimentdata

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

    def get_n_best_output_samples(self, nosamples: int) -> pd.DataFrame:
        """
        Get the n best output samples from the ExperimentData object.

        Parameters
        ----------
        nosamples : int
            Number of samples to retrieve.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the n best output samples.
        """
        df = self.output_data.n_best_samples(nosamples, self.output_data.names)
        return self.input_data.data.loc[df.index]

    def get_n_best_input_parameters_numpy(self, nosamples: int) -> np.ndarray:
        """
        Get the input parameters of the n best output samples from the ExperimentData object.

        Parameters
        ----------
        nosamples : int
            Number of samples to retrieve.

        Returns
        -------
        np.ndarray
            Numpy array containing the input parameters of the n best output samples.
        """
        return self.get_n_best_output_samples(nosamples).to_numpy()

    def get_input_data(self) -> pd.DataFrame:
        """
        Get the input data from the ExperimentData object.

        Returns
        -------
        pd.DataFrame
            DataFrame containing only the input data.
        """
        return self.input_data.data

    def get_output_data(self) -> pd.DataFrame:
        """
        Get the output data from the ExperimentData object.

        Returns
        -------
        pd.DataFrame
            DataFrame containing only the output data.
        """
        return self.output_data.data

    #                                                         Append or remove data
    # =============================================================================

    def add_new_input_column(self, name: str, parameter: Parameter) -> None:
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

    def add_new_output_column(self, name: str) -> None:
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
            status = OPEN
            self.output_data.add_empty_rows(len(input))
        else:
            status = FINISHED
            self.output_data.add_numpy_arrays(output)

        self.jobs.add(number_of_jobs=len(input), status=status)

    def add_design(self, design: Design) -> None:
        """
        Add a design to the ExperimentData object.

        Parameters
        ----------
        design : Design
            Design to add.
        """
        # Note: The index needs to be set but will not be used when adding the data!
        self.input_data.add(pd.DataFrame(design.input_data, index=[0]))
        self.output_data.add(pd.DataFrame(design.output_data, index=[0]))
        self.jobs.add(1, status=OPEN)

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

        self.output_data.fill_numpy_arrays(output)

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

    #                                                                        Design
    # =============================================================================

    def get_design(self, index: int) -> Design:
        """
        Gets the design at the given index.

        Parameters
        ----------
        index : int
            The index of the design to retrieve.

        Returns
        -------
        Design
            The design at the given index.
        """
        return Design(dict_input=self.input_data.get_data_dict(index),
                      dict_output=self.output_data.get_data_dict(index), jobnumber=index)

    def set_design(self, design: Design) -> None:
        """
        Sets the design at the given index.

        Parameters
        ----------
        design : Design
            The design to set.
        """
        for column, value in design.output_data.items():
            self.output_data.set_data(index=design.job_number, value=value, column=column)

        self.jobs.mark_as_finished(design._jobnumber)

    @_access_file
    def write_design(self, design: Design) -> None:
        """
        Sets the design at the given index.

        Parameters
        ----------
        design : Design
            The design to set.
        """
        self.set_design(design)

    def access_open_job_data(self) -> Design:
        """Get the data of the first available open job.

        Returns
        -------
        Design
            The Design object of the first available open job.
        """
        job_index = self.jobs.get_open_job()
        self.jobs.mark_as_in_progress(job_index)
        design = self.get_design(job_index)
        return design

    @_access_file
    def get_open_job_data(self) -> Design:
        """Get the data of the first available open job by
        accessing the ExperimenData on disk.

        Returns
        -------
        Design
            The Design object of the first available open job.
        """
        return self.access_open_job_data()

    #                                                                          Jobs
    # =============================================================================

    def set_error(self, index: int) -> None:
        """Mark the design at the given index as error.

        Parameters
        ----------
        index
            index of the design to mark as error
        """
        self.jobs.mark_as_error(index)
        self.output_data.set_data(index, value='ERROR')

    @_access_file
    def write_error(self, index: int):
        """Mark the design at the given index as error and write to ExperimentData file.

        Parameters
        ----------
        index
            index of the design to mark as error
        """
        self.set_error(index)

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

    #                                                            Run datageneration
    # =============================================================================

    def run(self, operation: _DesignCallable, mode: str = 'sequential', kwargs: dict = None) -> None:
        """Run any function over the entirery of the experiments

        Parameters
        ----------
        operation : DesignCallable
            function execution for every entry in the ExperimentData object
        mode, optional
            operational mode, by default 'sequential'
        kwargs, optional
            Any keyword arguments that need to be supplied to the function, by default None

        Raises
        ------
        TypeError
            Raised when the operation is not a callable function
        ValueError
            Raised when invalid parallelization mode is specified
        """
        if kwargs is None:
            kwargs = {}

        # Check if operation is a function
        if not callable(operation):
            raise TypeError("operation must be a function.")

        if mode.lower() == "sequential":
            return self._run_sequential(operation, kwargs)
        elif mode.lower() == "parallel":
            return self._run_multiprocessing(operation, kwargs)
        elif mode.lower() == "cluster":
            return self._run_cluster(operation, kwargs)
        else:
            raise ValueError("Invalid parallelization mode specified.")

    def _run_sequential(self, operation: _DesignCallable, kwargs: dict):
        """Run the operation sequentially

        Parameters
        ----------
        operation : DesignCallable
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
                design = self.access_open_job_data()
                logger.debug(f"Accessed design {design._jobnumber}")
            except NoOpenJobsError:
                logger.debug("No Open Jobs left")
                break

            try:

                # If kwargs is empty dict
                if not kwargs:
                    logger.info(f"Running design {design._jobnumber}")
                else:
                    logger.info(
                        f"Running design {design._jobnumber} with kwargs {kwargs}")

                _design = operation(design, **kwargs)  # no *args!
                self.set_design(_design)
            except Exception as e:
                error_msg = f"Error in design {design._jobnumber}: {e}"
                error_traceback = traceback.format_exc()
                logger.error(f"{error_msg}\n{error_traceback}")
                self.set_error(design._jobnumber)

    def _run_multiprocessing(self, operation: _DesignCallable, kwargs: dict):
        """Run the operation on multiple cores

        Parameters
        ----------
        operation : DesignCallable
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
                design = self.access_open_job_data()
                options.append(
                    ({'design': design, **kwargs},))
            except NoOpenJobsError:
                break

        def f(options: Dict[str, Any]) -> Any:
            logger.debug(f"Running design {options['design'].job_number}")
            return operation(**options)

        with mp.Pool() as pool:
            # maybe implement pool.starmap_async ?
            _designs: List[Design] = pool.starmap(f, options)

        for _design in _designs:
            self.set_design(_design)

    def _run_cluster(self, operation: _DesignCallable, kwargs: dict):
        """Run the operation on the cluster

        Parameters
        ----------
        operation : DesignCallable
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
                design = self.get_open_job_data()
            except NoOpenJobsError:
                logger.debug("No Open jobs left!")
                break

            try:
                _design = operation(design, **kwargs)
                self.write_design(_design)
            except Exception as e:
                error_msg = f"Error in design {design._jobnumber}: {e}"
                error_traceback = traceback.format_exc()
                logger.error(f"{error_msg}\n{error_traceback}")
                self.write_error(design._jobnumber)
                continue

        self = self.from_file(self.filename)
        # Remove the lockfile from disk
        Path(self.filename).with_suffix('.lock').unlink(missing_ok=True)
