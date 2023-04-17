#                                                                       Modules
# =============================================================================

# Standard
import errno
import functools
import json
import logging
import os
from copy import copy
from io import TextIOWrapper
from time import sleep
from typing import Any, Callable, List, Tuple, Type, Union

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
from .design import DesignSpace

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


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
        def wrapper_func(self, filename: str, *args, **kwargs) -> None:
            while True:
                try:
                    # Try to open the experimentdata file
                    with open(f"{filename}_data.csv", 'rb+') as file:
                        if os.name == 'nt':
                            msvcrt.locking(file.fileno(), msvcrt.LK_LOCK, 1)
                        else:
                            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)

                        # Load the experimentdata from the object
                        modified_experimentdata = ExperimentData.from_csv(filename=filename, text_io=file)
                        self.data = modified_experimentdata.data

                        # Do the operation
                        value = operation(self, filename, *args, **kwargs)

                        # Delete existing contents of file
                        file.seek(0, 0)
                        file.truncate()

                        # Write the data to disk
                        self.data.to_csv(file)

                    break
                except IOError as e:
                    # the file is locked by another process
                    if os.name == 'nt':
                        if e.errno == 13:
                            logging.info("The data file is currently locked by another process. "
                                         "Retrying in 1 second...")
                            sleep(sleeptime_sec)
                        elif e.errno == 2:  # File not found error
                            logging.info("The data file does not exist. Retrying in 1 second...")
                            sleep(sleeptime_sec)
                        else:
                            logging.info(f"An unexpected IOError occurred: {e}")
                            break
                    else:
                        if e.errno == errno.EAGAIN:
                            logging.info("The data file is currently locked by another process. "
                                         "Retrying in 1 second...")
                            sleep(sleeptime_sec)
                        elif e.errno == 2:  # File not found error
                            logging.info("The data file does not exist. Retrying in 1 second...")
                            sleep(sleeptime_sec)
                        else:
                            logging.info(f"An unexpected IOError occurred: {e}")
                            break
                except Exception as e:
                    # handle any other exceptions
                    logging.info(f"An unexpected error occurred: {e}")
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
        self.data = self.design.get_empty_dataframe()

    def __len__(self):
        """The len() method returns the number of datapoints"""
        return self.get_number_of_datapoints()

    def __iter__(self):
        self.current_index = 0
        return self

    def __next__(self):
        if self.current_index >= len(self):
            raise StopIteration
        else:
            index = self.data.index[self.current_index]
            current_value = [self.get_inputdata_by_index(
                index), self.get_outputdata_by_index(index)]
            self.current_index += 1
            return current_value

    @classmethod
    def from_csv(cls: Type['ExperimentData'], filename: str,
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
        # Load the design from a json string
        with open(f"{filename}_design.json") as f:
            design = DesignSpace.from_json(f.read())

        # Load the data from a csv
        if text_io is None:
            file = f"{filename}_data.csv"
        else:
            file = text_io
        data = pd.read_csv(file, header=[0, 1], index_col=0)

        # Create the experimentdata object
        experimentdata = cls(design=design)
        experimentdata.data = data
        return experimentdata

    @classmethod
    def from_json(cls: Type['ExperimentData'], json_string: str) -> 'ExperimentData':
        """
        Create an ExperimentData object from a JSON string.

        Parameters
        ----------
        json_string : str
            JSON string encoding the ExperimentData object.

        Returns
        -------
        ExperimentData
            The created ExperimentData object.
        """
        # Read JSON
        experimentdata_dict = json.loads(json_string)
        return cls.from_dict(experimentdata_dict)

    @classmethod
    def from_dict(cls: Type['ExperimentData'], experimentdata_dict: dict) -> 'ExperimentData':
        """
        Create an ExperimentData object from a dictionary.

        Parameters
        ----------
        experimentdata_dict : dict
            Dictionary representation of the information to construct the ExperimentData.

        Returns
        -------
        ExperimentData
            The created ExperimentData object.
        """
        # Read design from json_data_loaded
        new_design = DesignSpace.from_json(experimentdata_dict['design'])

        # Read data from json string
        new_data = pd.read_json(experimentdata_dict['data'])

        # Create tuples of indices
        columntuples = tuple(tuple(entry[1:-1].replace("'", "").split(', ')) for entry in new_data.columns.values)

        # Create MultiIndex object
        columnlabels = pd.MultiIndex.from_tuples(columntuples)

        # Overwrite columnlabels
        new_data.columns = columnlabels

        # Create
        new_experimentdata = cls(design=new_design)
        new_experimentdata.add(data=new_data)

        return new_experimentdata

    def select(self, indices: List[int]) -> 'ExperimentData':
        new_experimentdata = copy(self)
        new_experimentdata.data = self.data.iloc[indices].copy()
        return new_experimentdata

    def reset_data(self):
        """Reset the dataframe to an empty dataframe with the appropriate input and output columns"""
        self.__post_init__()

    def show(self):
        """Print the data to the console"""
        print(self.data)
        return

    def _store_textiowrapper(self, textio: TextIOWrapper):
        self.data.to_csv(textio)

    def store(self, filename: str, text_io: Union[TextIOWrapper, None] = None):
        """Store the ExperimentData to disk, with checking for a lock

        Parameters
        ----------
        filename
            filename of the files to store, without suffix
        """

        if os.name == 'nt':  # Windows
            # Open the data.csv file with a lock
            with open(f"{filename}_data.csv", 'w') as f:
                msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)
                self.data.to_csv(f"{filename}_data.csv")
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)

        else:  # Unix
            # Open the data.csv file with a lock
            with open(f"{filename}_data.csv", 'w') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                self.data.to_csv(f"{filename}_data.csv")
                fcntl.flock(f, fcntl.LOCK_UN)

        # convert design to json
        design_json = self.design.to_json()

        # write json to disk
        with open(f"{filename}_design.json", 'w') as outfile:
            outfile.write(design_json)

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
            return self.data['input'].loc[index].to_dict()
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
            return self.data['output'].loc[index].to_dict()
        except KeyError as e:
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
        try:
            self.data.at[index, 'output'] = value
            # self.data['output'].loc[index] = value
        except KeyError as e:
            raise KeyError('Index does not exist in dataframe!')

    @access_file()
    def write_outputdata_by_index(self, filename: str, index: int, value: Any):
        self.set_outputdata_by_index(index=index, value=value)

    def set_inputdata_by_index(self, index: int, value: Any):
        """
        Sets the input data at the given index to the given value.

        Parameters
        ----------
        index : int
            The index of the input data to set.
        value : Any
            The value to set the input data to.
        """
        try:
            self.data['input'].loc[index] = value
        except KeyError as e:
            raise KeyError('Index does not exist in dataframe!')

    @access_file()
    def write_inputdata_by_index(self, filename: str, index: int, value: Any):
        self.set_inputdata_by_index(index=index, value=value)

    def to_json(self) -> str:
        """
        Convert the ExperimentData object to a JSON string.

        Returns
        -------
        str
            JSON representation of the ExperimentData object.
        """
        args = {'design': self.design.to_json(),
                'data': self.data.to_json()}

        return json.dumps(args)

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert the ExperimentData object to a tuple of numpy arrays.

        Returns
        -------
        tuple
            A tuple containing two numpy arrays, the first one for input columns, and the second for output columns.
        """
        return self.get_input_data().to_numpy(), self.get_output_data().to_numpy()

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
        self.data = pd.concat([self.data, data], ignore_index=ignore_index)

        # Apparently you need to cast the types again
        # TODO: Breaks if values are NaN or infinite
        self.data = self.data.astype(
            self.design._cast_types_dataframe(self.design.input_space, "input"))
        self.data = self.data.astype(self.design._cast_types_dataframe(
            self.design.output_space, "output"))

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
        self.data[("output", label)] = output

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
        df = pd.DataFrame(np.hstack((input, output)),
                          columns=self.data.columns)
        self.add(df, ignore_index=True)

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

        self.data = self.data.iloc[:-number_of_rows]

    def get_input_data(self) -> pd.DataFrame:
        """
        Get the input data from the ExperimentData object.

        Returns
        -------
        pd.DataFrame
            DataFrame containing only the input data.
        """
        return self.data["input"]

    def get_output_data(self) -> pd.DataFrame:
        """
        Get the output data from the ExperimentData object.

        Returns
        -------
        pd.DataFrame
            DataFrame containing only the output data.
        """
        return self.data["output"]

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

        return self.data.nsmallest(n=nosamples, columns=self.design.get_output_names())

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
        fig, ax = plt.figure(), plt.axes()

        ax.scatter(self.data[("input", input_par1)],
                   self.data[("input", input_par2)], s=3)

        ax.set_xlabel(input_par1)
        ax.set_ylabel(input_par2)

        return fig, ax
