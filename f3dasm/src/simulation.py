from abc import ABC
from typing import Any, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcol

from f3dasm.src.data import Data


def from_data_to_numpy_array_benchmarkfunction(
    data: Data,
) -> np.ndarray:
    # Check if doe is in right format
    if not data.doe.is_single_objective_continuous():
        raise TypeError(
            "All inputs and outputs need to be continuous parameters and output single objective"
        )

    return data.get_input_data().to_numpy()


class Function(ABC):
    """Interface of a continuous benchmark function

    Args:
        noise (bool): inflict Gaussian noise on the output.
        seed (Any|int): value to seed the random generator (Default = None).
    """

    def __init__(self, noise: bool = False, seed: Any or int = None):
        self.noise = noise
        self.seed = seed

        # Set the seed
        if seed:
            self.set_seed(seed)

    @staticmethod
    def set_seed(seed) -> None:
        """Set the seed of the random generator"""
        np.random.seed(seed)

    def eval(self, input_x: np.ndarray or Data) -> np.ndarray:
        """Evaluate the objective function
        Args:
            input_x (np.ndarray | Data object): input to be evaluated

        Returns:
            np.ndarray: output of the objective function
        """
        # If the input is a Data object
        if isinstance(input_x, Data):
            x = from_data_to_numpy_array_benchmarkfunction(data=input_x)

        else:
            x = input_x

        x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
        if x.ndim == 1:
            x = np.reshape(x, (-1, len(x)))  # reshape into 2d array

        y = np.atleast_1d(self.f(x))
        # add noise
        if self.noise:
            y = self.add_noise(y)

        return y

    def add_noise(self, y: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to the output of the function

        Args:
            y (np.ndarray): output of the objective function

        Returns:
            np.ndarray: output of the objective function with added noise
        """
        sigma = 0.2  # Hard coded amount of noise
        y_noise = np.random.normal(loc=0.0, scale=abs(sigma * y), size=None)
        return y + y_noise

    def f(self, x) -> np.ndarray:
        """Compute the analytical output of the objective function"""
        raise NotImplementedError("Subclasses should implement this method.")

    def dfdx(self, x) -> np.ndarray:
        """ "Compute the gradient at a particular point in space"""
        pass  # TO DO: implement way to get gradient

    def plot(
        self, orientation: str = "3D", px: int = 300, domain: List = [0, 1]
    ):  # pragma: no cover
        """Generate a surface plot, either 2D or 3D, of the function

        Args:
            orientation (str, optional): Either "2D" or "3D" orientation. Defaults to "3D".
            px (int, optional): Number of points per dimension. Defaults to 300.
            domain (List, optional): Domain that needs to be plotted . Defaults to [0, 1].

        Returns:
            fig, ax: Figure and axis
        """
        x1 = np.linspace(domain[0], domain[1], num=px)
        x2 = np.linspace(domain[0], domain[1], num=px)
        X1, X2 = np.meshgrid(x1, x2)

        Y = np.zeros([len(X1), len(X1)])

        for i in range(len(X1)):
            for j in range(len(X1)):
                xy = np.array([X1[i, j], X2[i, j]])
                Y[i, j] = self.eval(xy)

        dx = dy = (domain[1] - domain[0]) / px
        x, y = domain[0] + dx * np.arange(Y.shape[1]), domain[0] + dy * np.arange(
            Y.shape[0]
        )
        xv, yv = np.meshgrid(x, y)

        fig = plt.figure(figsize=(7, 7), constrained_layout=True)
        if orientation == "2D":
            ax = plt.axes()
            ax.pcolormesh(xv, yv, Y, cmap="viridis", norm=mcol.LogNorm())
            # fig.colorbar(cm.ScalarMappable(norm=mcol.LogNorm(), cmap="viridis"), ax=ax)

        if orientation == "3D":
            ax = plt.axes(projection="3d", elev=50, azim=-50)

            ax.plot_surface(
                xv,
                yv,
                Y,
                rstride=1,
                cstride=1,
                edgecolor="none",
                alpha=0.8,
                cmap="viridis",
                norm=mcol.LogNorm(),
                zorder=1,
            )
            ax.set_zlabel("$f(X)$", fontsize=16)

        ax.set_xticks(np.linspace(domain[0], domain[1], num=11))
        ax.set_yticks(np.linspace(domain[0], domain[1], num=11))

        ax.set_xlabel("$X_{1}$", fontsize=16)
        ax.set_ylabel("$X_{2}$", fontsize=16)

        # ax.legend(fontsize="small", loc="lower right")

        return fig, ax
