#                                                                       Modules
# =============================================================================

# Standard
import os
from typing import Any, List, Tuple

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

    def reset_data(self):
        """Reset the dataframe to an empty dataframe with the appropriate input and output columns"""
        self.__post_init__()

    def show(self):
        """Print the data to the console"""
        print(self.data)
        return

    def store(self, filename: str):
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
            self.data['output'].loc[index] = value
        except KeyError as e:
            raise KeyError('Index does not exist in dataframe!')

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
