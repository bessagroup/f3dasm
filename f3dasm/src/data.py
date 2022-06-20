from dataclasses import dataclass

import pandas as pd
import matplotlib.pyplot as plt


@dataclass
class Data:
    """Class that contains data

    Args:
        data (DataFrame): data stored in a DataFrame
    """

    data: pd.DataFrame

    def plot(self, par1: str, par2: str) -> None:
        """Plot the data of two parameters in a figure

        Args:
            par1 (str): name of first parameter (x-axis)
            par2 (str): name of second parameter (y-axis)
        """
        fig, ax = plt.figure(), plt.axes()

        ax.scatter(self.data[par1], self.data[par2], s=3)

        ax.set_xlabel(par1)
        ax.set_ylabel(par2)

        fig.show()
