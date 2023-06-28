#                                                                       Modules
# =============================================================================

# Standard
import errno
import functools
import json
import os
from copy import deepcopy
from io import TextIOWrapper
from pathlib import Path
from time import sleep
from typing import (Any, Callable, Dict, Iterator, List, Protocol, Tuple, Type,
                    Union)

from .._logging import logger

# import msvcrt if windows, otherwise (Unix system) import fcntl
if os.name == 'nt':
    import msvcrt
else:
    import fcntl

# Third-party core
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Local
from ._data import Data
from ._jobqueue import JobQueue
from .design import DesignSpace

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


def access_file(sleeptime_sec: int = 1) -> Callable:
    """Wrapper for accessing a single resource with a file lock

    Parameters
    ----------
    sleeptime_sec, optional
        number of seconds to wait before trying to access resource again, by default 1

    Returns
    -------
    decorator
    """
    def decorator_func(operation: Callable) -> Callable:
        @functools.wraps(operation)
        def wrapper_func(self, *args, **kwargs) -> None:
            while True:
                try:
                    # Try to open the experimentdata file
                    logger.debug(f"Trying to open the data file: {self.filename}_data.csv")
                    with open(f"{self.filename}_data.csv", 'rb+') as file:
                        logger.debug("Opened file successfully")
                        if os.name == 'nt':
                            msvcrt.locking(file.fileno(), msvcrt.LK_LOCK, 1)
                        else:
                            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            logger.debug("Locked file successfully")

                        # Load the experimentdata from the object
                        self.data = Data.from_file(filename=self.filename, text_io=file)
                        logger.debug("Loaded data successfully")

                        # Load the jobs from disk
                        self.jobs = JobQueue.from_file(filename=f"{self.filename}_jobs")
                        logger.debug("Loaded jobs successfully")

                        # Do the operation
                        value = operation(self, *args, **kwargs)

                        # Delete existing contents of file
                        file.seek(0, 0)
                        file.truncate()

                        # Write the data to disk
                        self.data.store(filename=f"{self.filename}_data", text_io=file)
                        self.jobs.store(filename=f"{self.filename}_jobs")

                    break
                except IOError as e:
                    # the file is locked by another process
                    if os.name == 'nt':
                        if e.errno == 13:
                            logger.info("The data file is currently locked by another process. "
                                        "Retrying in 1 second...")
                            sleep(sleeptime_sec)
                        elif e.errno == 2:  # File not found error
                            logger.info("The data file does not exist. Retrying in 1 second...")
                            sleep(sleeptime_sec)
                        else:
                            logger.info(f"An unexpected IOError occurred: {e}")
                            break
                    else:
                        if e.errno == errno.EAGAIN:
                            logger.info("The data file is currently locked by another process. "
                                        "Retrying in 1 second...")
                            sleep(sleeptime_sec)
                        elif e.errno == 2:  # File not found error
                            logger.info("The data file does not exist. Retrying in 1 second...")
                            sleep(sleeptime_sec)
                        else:
                            logger.info(f"An unexpected IOError occurred: {e}")
                            break
                except Exception as e:
                    # handle any other exceptions
                    logger.info(f"An unexpected error occurred: {e}")
                    raise e
                    return

            return value

        return wrapper_func

    return decorator_func


class ExperimentData:
    """
    A class that contains data for experiments.
    """

    def __init__(self, design: DesignSpace):
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
        self.data = Data.from_design(self.design)
        self.jobs = JobQueue.from_data(self.data)
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

    @classmethod
    def from_file(cls: Type['ExperimentData'], filename: str = 'doe',
                  text_io: Union[TextIOWrapper, None] = None) -> 'ExperimentData':
        """Create an ExperimentData object from .csv and .json files.

        Parameters
        ----------
        filename : str
            Name of the file, excluding suffix.

        Returns
        -------
        ExperimentData
            ExperimentData object containing the loaded data.
        """
        # Create the experimentdata object
        design = DesignSpace.from_file(f"{filename}_design")
        experimentdata = cls(design=design)
        experimentdata.data = Data.from_file(f"{filename}_data", text_io)
        experimentdata.jobs = JobQueue.from_file(f"{filename}_jobs")
        experimentdata.filename = filename
        return experimentdata

    # create an alias of from_csv
    from_csv = from_file

    def select(self, indices: List[int]) -> 'ExperimentData':
        new_experimentdata = deepcopy(self)
        new_experimentdata.data.select(indices)
        new_experimentdata.jobs.select(indices)

        return new_experimentdata

    def reset_data(self):
        """Reset the dataframe to an empty dataframe with the appropriate input and output columns"""
        self.data.reset(self.design)
        self.jobs.reset()

    def show(self):
        """Print the data to the console"""
        print(self.data.data)
        return

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

        # # convert design to json
        # design_json = self.design.to_json()

        # # write json to disk
        # with open(f"{filename}_design.json", 'w') as outfile:
        #     outfile.write(design_json)

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

    @access_file()
    def get_open_job_data(self) -> Tuple[int, Dict[str, Any], Dict[str, Any]]:
        job_index = self.jobs.get_open_job()
        self.jobs.mark_as_in_progress(job_index)

        input_data = self.get_inputdata_by_index(job_index)
        output_data = self.get_outputdata_by_index(job_index)

        return job_index, input_data, output_data

    @access_file()
    def write_error(self, index: int):
        self.jobs.mark_as_error(index)
        self.set_outputdata_by_index(index, value='ERROR')

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert the ExperimentData object to a tuple of numpy arrays.

        Returns
        -------
        tuple
            A tuple containing two numpy arrays, the first one for input columns, and the second for output columns.
        """
        return self.data.to_numpy()

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

    def add_numpy_arrays(self, input: np.ndarray, output: np.ndarray):
        """
        Append a numpy array to the ExperimentData object.

        Parameters
        ----------
        input : np.ndarray
            2D numpy array to add to the input data.
        output : np.ndarray
            2D numpy array to add to the output data.
        """
        self.data.add_numpy_arrays(input, output)
        self.jobs.add(number_of_jobs=len(input))

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
