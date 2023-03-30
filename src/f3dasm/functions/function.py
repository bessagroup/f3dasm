"""
This module contains a base class for an analytical function that can be inherited
to create specific analytical functions.
The Function class is the base class that defines the interface for all analytical
functions. It can be called with an input vector to evaluate the function at that point.
"""
#                                                                       Modules
# =============================================================================

# Standard
from typing import Tuple, Union

# Third-party core
import autograd.numpy as np
import matplotlib.colors as mcol
import matplotlib.pyplot as plt

# Locals
from ..design.experimentdata import ExperimentData
from ..functions.adapters.augmentor import FunctionAugmentor

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class Function:
    def __init__(self, seed=None):
        """Class for an analytical function

        Parameters
        ----------
        seed, optional
            seed for the random number generator, by default None
        """
        self.augmentor = FunctionAugmentor()
        self.seed = seed

        self.set_seed(seed)

    def set_seed(self, seed):
        """Set the seed of the random number generator. By default the numpy generator

        Parameters
        ----------
        seed
            seed for the random number generator
        """
        if seed is None:
            return
        np.random.seed(seed)

    def __call__(self, input_x: Union[np.ndarray, ExperimentData]) -> np.ndarray:

        # If the input is a Data object
        if isinstance(input_x, ExperimentData):
            x = _from_data_to_numpy_array_benchmarkfunction(data=input_x)

        else:
            x = input_x

        x = np.atleast_2d(x)
        y = []
        for xi in x:
            xxi = self.augmentor.augment_input(xi)
            yi = self.evaluate(xxi)

            yyi = self.augmentor.augment_output(yi)
            y.append(yyi)

        # If the input is a Data object
        if isinstance(input_x, ExperimentData):
            input_x.add_output(np.array(y).reshape(-1, 1))

        return np.array(y).reshape(-1, 1)

    def _retrieve_original_input(self, x: np.ndarray):
        """Retrieve the original input vector if the input is augmented

        Parameters
        ----------
        x
            augmented input vector

        Returns
        -------
            original input vector
        """
        x = np.atleast_2d(x)
        xxi = self.augmentor.augment_reverse_input(x)
        return xxi

    def check_if_within_bounds(self, x: np.ndarray, bounds=np.ndarray) -> bool:
        """Check if the input vector is between the given scaling bounds

        Parameters
        ----------
        x
            input vector
        bounds
            boundaries for each dimension

        Returns
        -------
            boolean value whether the vector is within the boundaries
        """
        return ((bounds[:, 0] <= x) & (x <= bounds[:, 1])).all()

    def dfdx_legacy(self, x: np.ndarray, dx=1e-8) -> np.ndarray:
        """Compute the gradient at a particular point in space. Gradient is computed by central differences

        Parameters
        ----------
        x
            input vector

        Returns
        -------
            gradient
        """

        def central_differences(x: float, h: float):
            g = (self(x + h) - self(x - h)) / (2 * dx)
            return g.ravel().tolist()

        # dx = 1e-8

        grad = []
        for index, param in enumerate(x):
            # print(f"{index} {param}")
            h = np.zeros(x.shape)
            h[index] = dx
            grad.append(central_differences(x=param, h=h))

        grad = np.array(grad)
        return grad.ravel()

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Analytical expression to calculate the objective value
        To be inherited by subclasses

        Parameters
        ----------
        x
            input_vector

        Returns
        -------
            objective value(s)
        """
        ...

    def get_name(self) -> str:
        """Get the name of the function

        Returns
        -------
            name of the function
        """
        return self.__class__.__name__

    def _create_mesh(self, px: int, domain: np.ndarray):
        """Create mesh to use for plotting

        Parameters
        ----------
        px
            Number of points per dimension
        domain
            Domain that needes to be plotted

        Returns
        -------
            2D mesh used for plotting and another mesh
        """
        x1 = np.linspace(domain[0, 0], domain[0, 1], num=px)
        x2 = np.linspace(domain[1, 0], domain[1, 1], num=px)
        X1, X2 = np.meshgrid(x1, x2)

        Y = np.zeros([len(X1), len(X2)])

        for i in range(len(X1)):
            for j in range(len(X1)):
                xy = np.array([X1[i, j], X2[i, j]] + [0.0]
                              * (self.dimensionality - 2))
                Y[i, j] = self(xy)

        dx = (domain[0, 1] - domain[0, 0]) / px
        dy = (domain[1, 1] - domain[1, 0]) / px
        x = domain[0, 0] + dx * np.arange(Y.shape[0])
        y = domain[1, 0] + dy * np.arange(Y.shape[1])
        xv, yv = np.meshgrid(x, y)
        return xv, yv, Y

    def plot(
        self,
        orientation: str = "3D",
        px: int = 300,
        domain: np.ndarray = np.array([[0.0, 1.0], [0.0, 1.0]]),
        show: bool = True,
    ) -> Tuple[plt.Figure, plt.Axes]:  # pragma: no cover
        """Generate a surface plot, either 2D or 3D, of the function

        Parameters
        ----------
        orientation, optional
            Either 2D or 3D orientation
        px, optional
            Number of points per dimension
        domain, optional
            Domain that needs to be plotted
        show, optional
            Show the figure in interactive mode

        Returns
        -------
            matplotlib figure object and axes of the figure
        """
        if not show:
            plt.ioff()
        else:
            plt.ion()

        xv, yv, Y = self._create_mesh(px=px, domain=domain)

        fig = plt.figure(figsize=(7, 7), constrained_layout=True)
        if orientation == "2D":
            ax = plt.axes()
            ax.pcolormesh(xv, yv, Y, cmap="viridis",
                          norm=mcol.LogNorm())  # mcol.LogNorm()
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
                norm=mcol.LogNorm(),  # mcol.LogNorm()
                zorder=1,
            )
            ax.set_zlabel("$f(X)$", fontsize=16)

        ax.set_xticks(np.linspace(domain[0, 0], domain[0, 1], num=11))
        ax.set_yticks(np.linspace(domain[1, 0], domain[1, 1], num=11))

        ax.set_xlabel("$X_{0}$", fontsize=16)
        ax.set_ylabel("$X_{1}$", fontsize=16)

        # ax.legend(fontsize="small", loc="lower right")
        if not show:
            plt.close(fig)

        return fig, ax

    def plot_data(
        self, data: ExperimentData, px: int = 300, domain: np.ndarray = np.array([[0.0, 1.0], [0.0, 1.0]]),
        numsamples=None, arrow=False
    ) -> Tuple[plt.Figure, plt.Axes]:  # pragma: no cover
        """Create a 2D contout plot with the datapoints as scatter

        Parameters
        ----------
        data
            Data object containing samples
        px, optional
            number of pixels on each axis
        domain, optional
            domain that needs to be plotted

        Returns
        -------
            matplotlib figure and axes
        """
        fig, ax = self.plot(orientation="2D", px=px, domain=domain)
        x1 = data.get_input_data().iloc[:, 0]
        x2 = data.get_input_data().iloc[:, 1]
        ax.scatter(
            x=x1,
            y=x2,
            s=10,
            c=np.linspace(0, 1, len(x1)),
            cmap="Blues",
            edgecolors="black",
        )
        if arrow:
            for p_index in range(len(x1)-1):
                dx = (x1[p_index+1] - x1[p_index])
                dy = (x2[p_index+1] - x2[p_index])
                length = 1/np.sqrt(dx**2 + dy**2)
                ax.arrow(x=x1[p_index], y=x2[p_index], dx=dx*.1*length, dy=dy*.1*length, shape='full',
                         length_includes_head=True)

        # Mark selected point
        if numsamples is not None:
            x_selected = data.get_input_data().iloc[numsamples]
            ax.scatter(x=x_selected[0], y=x_selected[1], s=25, c="cyan",
                       marker="*", edgecolors="cyan")

        # Mark last point
        x_last = data.get_input_data().iloc[-1]
        ax.scatter(x=x_last[0], y=x_last[1], s=25, c="magenta",
                   marker="*", edgecolors="magenta")

        # Best point
        x1_best = data.get_n_best_output_samples(nosamples=1).iloc[:, 0]
        x2_best = data.get_n_best_output_samples(nosamples=1).iloc[:, 1]
        ax.scatter(x=x1_best, y=x2_best, s=25, c="red",
                   marker="*", edgecolors="red")
        return fig, ax


def _from_data_to_numpy_array_benchmarkfunction(
    data: ExperimentData,
) -> np.ndarray:
    """Check if doe is in right format"""
    if not data.design.is_single_objective_continuous():
        raise TypeError(
            "All inputs and outputs need to be continuous parameters and output single objective")

    return data.get_input_data().to_numpy()
