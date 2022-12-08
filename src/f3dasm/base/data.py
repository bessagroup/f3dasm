#                                                                       Modules
# =============================================================================

# Standard
from typing import Tuple

# Third-party
import autograd.numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Local
from ..base.design import DesignSpace

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class Data:
    def __init__(self, design: DesignSpace):
        """Class that contains data

        Parameters
        ----------
        data
            data stored in a DataFrame
        design
            designspace
        """
        self.design = design
        self.__post_init__()

    def __post_init__(self):
        self.data = self.design.get_empty_dataframe()

    def reset_data(self):
        """Reset the dataframe to an empty dataframe with the appropriate input and output columns"""
        self.__post_init__()

    def show(self):
        """Print the data to the console"""
        print(self.data)
        return

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Convert the data to a tuple numpy arrays

        Returns
        -------
            tuple of np.ndarrays: first of input columns, second one of output columns
        """
        return self.get_input_data().to_numpy(), self.get_output_data().to_numpy()

    def add(self, data: pd.DataFrame, ignore_index: bool = False):
        """Add data

        Parameters
        ----------
        data
            data to append
        ignore_index, optional
            whether to ignore the indices of the appended dataframe
        """
        self.data = pd.concat([self.data, data], ignore_index=ignore_index)

        # Apparently you need to cast the types again
        # TODO: Breaks if values are NaN or infinite
        self.data = self.data.astype(
            self.design._cast_types_dataframe(self.design.input_space, "input"))
        self.data = self.data.astype(self.design._cast_types_dataframe(
            self.design.output_space, "output"))

    def add_output(self, output: np.ndarray, label: str = "y"):
        """Add a numpy array to the output column of the dataframe

        Parameters
        ----------
        output
            Output data
        label, optional
            label of the output column to add to
        """
        self.data[("output", label)] = output

    def add_numpy_arrays(self, input: np.ndarray, output: np.ndarray):
        """Append a numpy array to the datafram

        Parameters
        ----------
        input
            2d numpy array added to input data
        output
            2d numpy array added to output data
        """
        df = pd.DataFrame(np.hstack((input, output)),
                          columns=self.data.columns)
        self.add(df, ignore_index=True)

    def remove_rows_bottom(self, number_of_rows: int):
        """Remove a number of rows from the end of the Dataframe

        Parameters
        ----------
        number_of_rows
            number of rows to remove from the bottom
        """
        if number_of_rows == 0:
            return  # Don't do anything if 0 rows need to be removed

        self.data = self.data.iloc[:-number_of_rows]

    def get_input_data(self) -> pd.DataFrame:
        """Get the input data

        Returns
        -------
            DataFrame containing only the input data
        """
        return self.data["input"]

    def get_output_data(self) -> pd.DataFrame:
        """Get the output data

        Returns
        -------
            DataFrame containing only the output data
        """
        return self.data["output"]

    def get_n_best_output_samples(self, nosamples: int) -> pd.DataFrame:
        """Returns the n lowest rows of the dataframe. Values are compared to the output columns

        Parameters
        ----------
        nosamples :
            number of samples

        Returns
        -------
        pd.DataFrame
            Dataframe containing the n best samples
        """

        return self.data.nsmallest(n=nosamples, columns=self.design.get_output_names())

    def get_n_best_input_parameters_numpy(self, nosamples: int) -> np.ndarray:
        """Returns the input vector in numpoy array format of the n best samples

        Parameters
        ----------
        nosamples
            number of samples

        Returns
        -------
            numpy array containing the n best input parameters
        """
        return self.get_n_best_output_samples(nosamples)["input"].to_numpy()

    def get_number_of_datapoints(self) -> int:
        """Get the total number of datapoints

        Returns
        -------
            total number of datapoints
        """
        return len(self.data)

    def plot(self, input_par1: str = "x0", input_par2: str = "x1") -> Tuple[plt.Figure, plt.Axes]:
        """Plot the data of two parameters in a figure

        Parameters
        ----------
        input_par1
            name of first parameter (x-axis)
        input_par2
            name of second parameter (x-axis)

        Returns
        -------
            Matplotlib figure and axes
        """
        fig, ax = plt.figure(), plt.axes()

        ax.scatter(self.data[("input", input_par1)],
                   self.data[("input", input_par2)], s=3)

        ax.set_xlabel(input_par1)
        ax.set_ylabel(input_par2)

        return fig, ax

    def plot_pairs(self):
        """
        Plot a matrix of 2D plots that visualize the spread of the samples for each dimension.
        Requires seaborn to be installed.
        """
        import seaborn as sb

        sb.pairplot(data=self.get_input_data())
