import autograd.numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass, field

from ..base.design import DesignSpace


@dataclass
class Data:
    """Class that contains data

    Args:
        data (DataFrame): data stored in a DataFrame
    """

    design: DesignSpace
    data: pd.DataFrame = field(init=False)

    def __post_init__(self):
        self.data = self.design.get_empty_dataframe()

    def reset_data(self) -> None:
        """Reset the dataframe to an empty dataframe with the appropriate input and output columns"""
        self.__post_init__()

    def show(self) -> None:
        """Print the data to the console"""
        print(self.data)
        return

    def add(self, data: pd.DataFrame, ignore_index: bool = False) -> None:
        """Add data

        Args:
            data (pd.DataFrame): data to append
        """
        self.data = pd.concat([self.data, data], ignore_index=ignore_index)

        # Apparently you need to cast the types again
        # TODO: Breaks if values are NaN or infinite
        self.data = self.data.astype(self.design._cast_types_dataframe(self.design.input_space, "input"))
        self.data = self.data.astype(self.design._cast_types_dataframe(self.design.output_space, "output"))

    def add_output(self, output: np.ndarray, label: str = "y") -> None:
        """Add a numpy array to the output column of the dataframe

        Args:
            output (np.ndarray): Output data
            label (str): label of the output column to add to
        """
        self.data[("output", label)] = output

    def add_numpy_arrays(self, input: np.ndarray, output: np.ndarray) -> None:
        """Append a numpy array to the dataframe

        Args:
            input (np.ndarray): 2d numpy array added to input data
            output (np.ndarray): 2d numpy array added to output data
        """
        df = pd.DataFrame(np.hstack((input, output)), columns=self.data.columns)
        self.add(df, ignore_index=True)

    def remove_rows_bottom(self, number_of_rows: int) -> None:
        """Remove a number of rows from the end of the Dataframe

        Args:
            number_of_rows (int): number of rows to remove from the bottom
        """
        if number_of_rows == 0:
            return  # Don't do anything if 0 rows need to be removed

        self.data = self.data.iloc[:-number_of_rows]

    def get_input_data(self) -> pd.DataFrame:
        """Get the input data

        Returns:
            pd.DataFrame: DataFrame containing only the input data
        """
        return self.data["input"]

    def get_output_data(self) -> pd.DataFrame:
        """Get the output data

        Returns:
            pd.DataFrame: DataFrame containing only the output data
        """
        return self.data["output"]

    def get_n_best_output_samples(self, nosamples: int) -> pd.DataFrame:
        """Returns the n lowest rows of the dataframe. Values are compared to the output columns

        Args:
            nosamples (int): number of samples

        Returns:
            pd.DataFrame: DataFrame containing the n best samples
        """
        return self.data.nsmallest(n=nosamples, columns=self.design.get_output_names())

    def get_n_best_input_parameters_numpy(self, nosamples: int) -> np.ndarray:
        """Returns the input vectors in numpy array format of the n best samples

        Args:
            nosamples (int): number of samples

        Returns:
            np.ndarray: numpy array containing the n best input parameters
        """
        return self.get_n_best_output_samples(nosamples)["input"].to_numpy()

    def get_number_of_datapoints(self) -> int:
        """Get the total number of datapoints

        Returns:
            int: total number of datapoints
        """
        return len(self.data)

    def plot(self, input_par1: str, input_par2: str = None) -> None:
        """Plot the data of two parameters in a figure

        Args:
            input_par1 (str): name of first parameter (x-axis)
            input_par2 (str): name of second parameter (x-axis)
        """
        fig, ax = plt.figure(), plt.axes()

        ax.scatter(self.data[("input", input_par1)], self.data[("input", input_par2)], s=3)

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
