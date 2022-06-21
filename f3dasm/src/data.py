from dataclasses import dataclass

import pandas as pd
import matplotlib.pyplot as plt

from f3dasm.src.designofexperiments import DoE


@dataclass
class Data:
    """Class that contains data

    Args:
        data (DataFrame): data stored in a DataFrame
    """

    data: pd.DataFrame

    def set_with_doe(self, doe: DoE) -> None:
        """set the design space of the data with a DoE object

        Args:
            doe (DoE): design of experiments

        Returns:
            pd.DataFrame: empty dataframe with columns
        """
        columns = [
            ["input"] * doe.getNumberOfInputParameters()
            + ["output"] * doe.getNumberOfOutputParameters(),
            [s.name for s in doe.input_space] + [s.name for s in doe.output_space],
        ]
        self.data = pd.DataFrame(columns=columns)

    def append(self, data: pd.DataFrame) -> None:
        """Add data

        Args:
            data (pd.DataFrame): data to append
        """
        self.data = self.data.append(data)

    def plot(
        self, input_par1: str, input_par2: str = None, output_par: str = None
    ) -> None:
        """Plot the data of two parameters in a figure

        Args:
            input_par1 (str): name of first parameter (x-axis)
            input_par2 (str): name of second parameter (x-axis)
            output_par (str): name of output parameter (y-axis)
        """
        fig, ax = plt.figure(), plt.axes()

        ax.scatter(
            self.data[("input", input_par1)], self.data[("input", input_par2)], s=3
        )

        ax.set_xlabel(input_par1)
        ax.set_ylabel(input_par2)

        fig.show()
