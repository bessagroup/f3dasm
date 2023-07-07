#                                                                       Modules
# =============================================================================

# Standard
import errno
import functools
import json
import os
import sys
import traceback
from copy import deepcopy
from io import TextIOWrapper
from pathlib import Path
from time import sleep

if sys.version_info < (3, 8):
    from typing_extensions import Protocol
else:
    from typing import Protocol

from typing import Any, Callable, Dict, Iterator, List, Tuple, Type, Union

# import msvcrt if windows, otherwise (Unix system) import fcntl
if os.name == 'nt':
    import msvcrt
else:
    import fcntl

# Third-party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from pathos.helpers import mp

# Local
from ..logger import logger
from ._access_file import access_file
from ._data import _Data
from ._jobqueue import NoOpenJobsError, _JobQueue
from .design import Design
from .domain import Domain

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class Sampler(Protocol):
    def get_samples(numsamples: int) -> 'ExperimentData':
        ...


class ExperimentData:
    """
    A class that contains data for experiments.
    """

    def __init__(self, design: Domain):
        """
        Initializes an instance of ExperimentData.

        Parameters
        ----------
        design : DesignSpace
            A DesignSpace object defining the input and output spaces of the experiment.
        """
        self.design = design
        self.__post_init__()

    def __post_init__(self):
        """Initializes an empty DataFrame with the appropriate input and output columns."""
        self.data = _Data.from_design(self.design)
        self.jobs = _JobQueue.from_data(self.data)
        self.filename = 'doe'

    def __len__(self):
        """The len() method returns the number of datapoints"""
        return self.get_number_of_datapoints()

    def __iter__(self) -> Iterator[Tuple[Dict[str, Any]]]:
        self.current_index = 0
        return self

    def __next__(self):
        self.data.__next__()

    def _repr_html_(self) -> str:
        return self.data._repr_html_()

    #                                                      Alternative Constructors
    # =============================================================================

    @classmethod
    def from_file(cls: Type['ExperimentData'], filename: str = 'doe',
                  text_io: Union[TextIOWrapper, None] = None) -> 'ExperimentData':
        """Create an ExperimentData object from .csv and .json files.

        Parameters
        ----------
        filename : str
            Name of the file, excluding suffix.
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
                raise FileNotFoundError(f"Cannot find the file {filename}_data.csv.")

            return cls._from_file_attempt(filename_with_path, text_io)

    @classmethod
    def _from_file_attempt(cls: Type['ExperimentData'], filename: str,
                           text_io: Union[TextIOWrapper, None]) -> 'ExperimentData':
        try:
            design = Domain.from_file(f"{filename}_design")
            experimentdata = cls(design=design)
            experimentdata.data = _Data.from_file(f"{filename}_data", text_io)
            experimentdata.jobs = _JobQueue.from_file(f"{filename}_jobs")
            experimentdata.filename = filename
            return experimentdata
        except FileNotFoundError:
            raise FileNotFoundError(f"Cannot find the file {filename}_data.csv.")

    @classmethod
    def from_sampling(cls, sampler: Sampler) -> 'ExperimentData':
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

        return sampler.get_samples()

    @classmethod
    def from_yaml(cls, config: DictConfig) -> 'ExperimentData':

        # Option 1: From existing ExperimentData files
        if config.experimentdata.existing_data_path:
            data = cls.from_file(filename=config.experimentdata.data)

        # Option 2: Sample from the designspace
        else:
            sampler = Sampler.from_yaml(config)
            data = sampler.get_samples(config.experimentdata.number_of_samples)

        return data

    def select(self, indices: List[int]) -> 'ExperimentData':
        new_experimentdata = deepcopy(self)
        new_experimentdata.data.select(indices)
        new_experimentdata.jobs.select(indices)

        return new_experimentdata

    #                                                               Storage Methods
    # =============================================================================

    def store(self, filename: str = None, text_io: Union[TextIOWrapper, None] = None):
        """Store the ExperimentData to disk, with checking for a lock

        Parameters
        ----------
        filename
            filename of the files to store, without suffix
        """
        if filename is None:
            filename = self.filename

        self.data.store(f"{filename}_data")
        self.jobs.store(f"{filename}_jobs")
        self.design.store(f"{filename}_design")

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert the ExperimentData object to a tuple of numpy arrays.

        Returns
        -------
        tuple
            A tuple containing two numpy arrays, the first one for input columns, and the second for output columns.
        """
        return self.data.to_numpy()

    def run(self, operation: Callable, mode: str = 'sequential', kwargs: dict = None):

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

    def reset_data(self):
        """Reset the dataframe to an empty dataframe with the appropriate input and output columns"""
        self.data.reset(self.design)
        self.jobs.reset()

    def get_inputdata_by_index(self, index: int) -> dict:
        """
        Gets the input data at the given index.

        Parameters
        ----------
        index : int
            The index of the input data to retrieve.

        Returns
        -------
        dict
            A dictionary containing the input data at the given index.
        """
        try:
            return self.data.get_inputdata_dict(index)
        except KeyError as e:
            raise KeyError('Index does not exist in dataframe!')

    def get_outputdata_by_index(self, index: int) -> dict:
        """
        Gets the output data at the given index.

        Parameters
        ----------
        index : int
            The index of the output data to retrieve.

        Returns
        -------
        dict
            A dictionary containing the output data at the given index.
        """
        try:
            return self.data.get_outputdata_dict(index)
        except KeyError:
            raise KeyError('Index does not exist in dataframe!')

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
        return self.data.get_design(index)

    def set_design(self, design: Design) -> None:
        """
        Sets the design at the given index.

        Parameters
        ----------
        design : Design
            The design to set.
        """
        self.data.set_design(design)
        self.jobs.mark_as_finished(design._jobnumber)

    @access_file()
    def write_design(self, design: Design) -> None:
        """
        Sets the design at the given index.

        Parameters
        ----------
        design : Design
            The design to set.
        """
        self.set_design(design)

    def set_outputdata_by_index(self, index: int, value: Any):
        """
        Sets the output data at the given index to the given value.

        Parameters
        ----------
        index : int
            The index of the output data to set.
        value : Any
            The value to set the output data to.
        """
        self.data.set_outputdata(index, value)
        self.jobs.mark_as_finished(index)

    @access_file()
    def write_outputdata_by_index(self, index: int, value: Any):
        self.set_outputdata_by_index(index=index, value=value)

    def set_inputdata_by_index(self, index: int, value: Any, column: str = 'input'):
        """
        Sets the input data at the given index to the given value.

        Parameters
        ----------
        index : int
            The index of the input data to set.
        value : Any
            The value to set the input data to.
        """
        self.data.set_inputdata(index, value, column)

    @access_file()
    def write_inputdata_by_index(self, index: int, value: Any, column: str = 'input'):
        self.set_inputdata_by_index(index=index, value=value, column=column)

    def access_open_job_data(self) -> Design:
        job_index = self.jobs.get_open_job()
        self.jobs.mark_as_in_progress(job_index)
        design = self.get_design(job_index)
        return design

    @access_file()
    def get_open_job_data(self) -> Design:
        return self.access_open_job_data()

    def set_error(self, index: int):
        self.jobs.mark_as_error(index)
        self.set_outputdata_by_index(index, value='ERROR')

    @access_file()
    def write_error(self, index: int):
        self.set_error(index)

    @access_file()
    def is_all_finished(self) -> bool:
        return self.jobs.is_all_finished()

    def add(self, data: pd.DataFrame, ignore_index: bool = False):
        """
        Append data to the ExperimentData object.

        Parameters
        ----------
        data : pd.DataFrame
            Data to append.
        ignore_index : bool, optional
            Whether to ignore the indices of the appended dataframe.
        """
        self.data.add(data)

        # Apparently you need to cast the types again
        # TODO: Breaks if values are NaN or infinite
        self.data.data = self.data.data.astype(
            self.design._cast_types_dataframe(self.design.input_space, "input"))
        self.data.data = self.data.data.astype(self.design._cast_types_dataframe(
            self.design.output_space, "output"))

        self.jobs.add(number_of_jobs=len(data))

    def add_output(self, output: np.ndarray, label: str = "y"):
        """
        Append a numpy array to the output column of the ExperimentData object.

        Parameters
        ----------
        output : np.ndarray
            Output data to append.
        label : str, optional
            Label of the output column to add to.
        """
        self.data.add_output(output, label)

    def add_numpy_arrays(self, input: np.ndarray, output: Union[np.ndarray, None] = None):
        """
        Append a numpy array to the ExperimentData object.

        Parameters
        ----------
        input : np.ndarray
            2D numpy array to add to the input data.
        output : np.ndarray
            2D numpy array to add to the output data.
        """
        if output is not None:
            status = 'finished'
        else:
            status = 'open'

        self.data.add_numpy_arrays(input, output)
        self.jobs.add(number_of_jobs=len(input), status=status)

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
        indices = self.data.data.index[-number_of_rows:]

        # remove the indices rows_to_remove from data.data
        self.data.remove(indices)
        self.jobs.remove(indices)

    def get_input_data(self) -> pd.DataFrame:
        """
        Get the input data from the ExperimentData object.

        Returns
        -------
        pd.DataFrame
            DataFrame containing only the input data.
        """
        return self.data.get_inputdata()

    def get_output_data(self) -> pd.DataFrame:
        """
        Get the output data from the ExperimentData object.

        Returns
        -------
        pd.DataFrame
            DataFrame containing only the output data.
        """
        return self.data.get_outputdata()

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
        return self.data.n_best_samples(nosamples, self.design.get_output_names())

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
        return self.get_n_best_output_samples(nosamples)["input"].to_numpy()

    def get_number_of_datapoints(self) -> int:
        """
        Get the total number of datapoints in the ExperimentData object.

        Returns
        -------
        int
            Total number of datapoints.
        """
        return len(self.data)

    def plot(self, input_par1: str = "x0", input_par2: str = "x1") -> Tuple[plt.Figure, plt.Axes]:
        """Plot the data of two parameters in a figure

        Parameters
        ----------
        input_par1: str, optional
            name of first parameter (x-axis)
        input_par2: str, optional
            name of second parameter (x-axis)

        Returns
        -------
        tuple
            A tuple containing the matplotlib figure and axes
        """
        fig, ax = self.data.plot()

        return fig, ax

    #                                                               Private methods
    # =============================================================================

    def _run_sequential(self, operation: Callable, kwargs: dict):
        while True:
            try:
                design = self.access_open_job_data()
                logger.debug(f"Accessed design {design._jobnumber}")
            except NoOpenJobsError:
                logger.debug("No Open Jobs left")
                break

            try:
                logger.info(
                    f"Running design {design._jobnumber} with kwargs {kwargs}")
                _design = operation(design, **kwargs)  # no *args!
                self.set_design(_design)
            except Exception as e:
                error_msg = f"Error in design {design._jobnumber}: {e}"
                error_traceback = traceback.format_exc()
                logger.error(f"{error_msg}\n{error_traceback}")
                self.set_error(design._jobnumber)

    def _run_multiprocessing(self, operation: Callable, kwargs: dict):
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
                return operation(**options)

            with mp.Pool() as pool:
                # maybe implement pool.starmap_async ?
                _designs: List[Design] = pool.starmap(f, options)

            for _design in _designs:
                self.set_design(_design)

    def _run_cluster(self, operation: Callable, kwargs: dict):
        # Retrieve the updated experimentdata object from disc
        try:
            self = self.from_file(self.filename)
        except FileNotFoundError:  # If not found, store current
            self.store()
            # _data = self.from_file(self.filename)

        while True:
            try:
                design = self.get_open_job_data()
            except NoOpenJobsError:
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
