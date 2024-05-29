"""
This module contains a base class for an analytical function that
 can be inherited
to create specific analytical functions.
The Function class is the base class that defines the interface for
 all analytical
functions. It can be called with an input vector to evaluate the
 function at that point.
"""
#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
from typing import Optional, Protocol, Tuple

# Third-party core
import autograd.numpy as np
import matplotlib.colors as mcol
import matplotlib.pyplot as plt
from autograd import grad
from autograd.numpy.numpy_boxes import ArrayBox

from ...design.domain import Domain
# Locals
from ...experimentdata.experimentsample import _experimentsample_factory
from ..datagenerator import DataGenerator
from ..functions.adapters.augmentor import FunctionAugmentor

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class ExperimentSample(Protocol):
    def __setitem__(self, key: str, value: np.ndarray):
        ...

    def to_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        ...


class Function(DataGenerator):
    def __init__(self, seed: Optional[int] = None):
        """Class for an analytical function

        Parameters
        ----------
        seed, optional
            seed for the random number generator, by default None
        """
        self.augmentor = FunctionAugmentor()
        self.seed = seed
        self.grad = grad(self.__call__)

        self.set_seed(seed)

    def set_seed(self, seed):
        """Set the seed of the random number generator.
         By default the numpy generator

        Parameters
        ----------
        seed
            seed for the random number generator
        """
        if seed is None:
            return
        np.random.seed(seed)

    def __call__(self, input_x: np.ndarray) -> np.ndarray:
        x = input_x

        x = np.atleast_2d(x)
        y = []
        for xi in x:
            xxi = self.augmentor.augment_input(xi)
            yi = self.evaluate(xxi)

            yyi = self.augmentor.augment_output(yi)
            y.append(yyi)

        return np.array(y).reshape(-1, 1)

    def execute(self, experiment_sample: ExperimentSample) -> ExperimentSample:
        x, _ = experiment_sample.to_numpy()

        if isinstance(x, ArrayBox):
            x = x._value
            if isinstance(x, ArrayBox):
                x = x._value
        y = np.nan_to_num(self(x), nan=np.nan)
        experiment_sample["y"] = float(y.ravel().astype(np.float64))
        return experiment_sample

    def _run(
            self, experiment_sample: ExperimentSample | np.ndarray,
            domain: Optional[Domain] = None, **kwargs) -> ExperimentSample:
        _experiment_sample = _experimentsample_factory(
            experiment_sample=experiment_sample, domain=domain)
        return self.execute(_experiment_sample)

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
        """Compute the gradient at a particular point in space.
         Gradient is computed by central differences

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
            h = np.zeros(x.shape)
            h[index] = dx
            grad.append(central_differences(x=param, h=h))

        grad = np.array(grad)
        return grad.ravel()

    def dfdx(self, x: np.ndarray) -> np.ndarray:
        # check if the object has a 'custom_grad' method
        if hasattr(self, 'error_autograd'):
            if self.error_autograd:
                return self.dfdx_legacy(x)

        return self.grad(x)

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
        ax: plt.Axes = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        # TODO: orientation string is case sensitive!
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

        # Shift Y values to avoid log(0)
        # Add a small constant to avoid log(0)
        Y_shifted = Y - np.min(Y) + 1e-6

        fig = plt.figure(figsize=(7, 7), constrained_layout=True)
        if orientation == "2D":
            if ax is None:
                ax = plt.axes()
            ax.pcolormesh(xv, yv, Y_shifted, cmap="viridis",
                          norm=mcol.LogNorm())

        if orientation == "3D":
            if ax is None:
                ax = plt.axes(projection="3d", elev=50, azim=-50)

            ax.plot_surface(
                xv,
                yv,
                Y_shifted,
                rstride=1,
                cstride=1,
                edgecolor="none",
                alpha=0.9,
                cmap="viridis",
                # norm=mcol.LogNorm(),  # mcol.LogNorm()
                zorder=1,
            )
            ax.set_zlabel("$f(X)$", fontsize=16)

        ax.set_xticks(np.linspace(domain[0, 0], domain[0, 1], num=11))
        ax.set_yticks(np.linspace(domain[1, 0], domain[1, 1], num=11))

        ax.set_xlabel("$X_{0}$", fontsize=16)
        ax.set_ylabel("$X_{1}$", fontsize=16)

        ax.set_xlim(domain[0, 0], domain[0, 1])
        ax.set_ylim(domain[1, 0], domain[1, 1])

        if not show:
            plt.close(fig)

        return fig, ax
