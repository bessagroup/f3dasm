from abc import ABC
from dataclasses import dataclass
from typing import Any
import autograd.numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
import numdifftools as nd
from scipy.stats import special_ortho_group

from ..base.data import Data
from ..base.utils import (
    _from_data_to_numpy_array_benchmarkfunction,
    _scale_vector,
    _descale_vector,
    _rotate_vector,
)


@dataclass
class Function(ABC):
    """Interface of a continuous benchmark function

    Args:
        noise (bool): inflict Gaussian noise on the output.
        seed (Any|int): value to seed the random generator (Default = None).
        dimensionality (int): input dimension
        scale_bounds (Any|List[float]): array containing the lower and upper bound of the scaling factor of the input data (Default = [0.0, 1.0])
        input_domain (np.ndarray): array containing the lower and upper bound of the input domain of the original function (Default = [0.0, 1.0])
    """

    noise: Any or float = None
    seed: Any or int = None
    dimensionality: int = 2
    scale_bounds: np.ndarray = np.tile([0.0, 1.0], (dimensionality, 1))
    input_domain: np.ndarray = np.tile([0.0, 1.0], (dimensionality, 1))

    def __post_init__(self):

        if self.seed:
            self.set_seed(self.seed)

        self._set_parameters()
        self.offset = np.zeros(self.dimensionality)
        # self.rotation_matrix = np.identity(self.dimensionality)
        # self.rotation_point = np.zeros(self.dimensionality)

        self._create_offset()
        # self._create_rotation_point()
        # self._create_rotation()

    def set_seed(self, seed: int) -> None:
        """Set the numpy seed of the random generator"""
        self.seed = seed
        np.random.seed(seed)

    def check_if_within_bounds(self, x: np.ndarray) -> bool:
        """Check if the input vector is between the given scaling bounds

        Args:
            x (np.ndarray): input vector

        Returns:
            bool: whether the vector is within the boundaries
        """
        return ((self.scale_bounds[:, 0] <= x) & (x <= self.scale_bounds[:, 1])).all()

    def __call__(self, input_x: np.ndarray or Data) -> np.ndarray or Data:
        """Evaluate the objective function
        Args:
            input_x (np.ndarray | Data object): input to be evaluated

        Returns:
            np.ndarray: output of the objective function
        """
        # If the input is a Data object
        if isinstance(input_x, Data):
            x = _from_data_to_numpy_array_benchmarkfunction(data=input_x)

        else:
            x = input_x

        # x = self._from_input_to_scaled(x)

        y = np.atleast_1d(self.f(x))

        # add noise
        if self.noise not in [None, "None"]:
            y = self._add_noise(y)

        # If the input is a Data object
        if isinstance(input_x, Data):
            input_x.add_output(y)
            # return input_x

        return y

    def f(self, x) -> np.ndarray:
        """Analytical function of the objective function. Needs to be implemented by inhereted class"""
        raise NotImplementedError("Subclasses should implement this method.")

    def dfdx(self, x: np.ndarray) -> np.ndarray:
        """Compute the gradient at a particular point in space. Gradient is computed by numdifftools.

        Args:
            x (np.ndarray): input vector

        Returns:
            np.ndarray: gradient
        """
        # TODO : fix the output shape (now it is shape=(dim*samples+1,), should be shape=(samples,1))
        grad = nd.Gradient(self)
        x = self._reshape_input(x)
        output = np.empty(shape=(1, len(x[0, :])))
        for i in range(len(x)):
            output = np.r_[output, np.atleast_2d(grad(np.atleast_2d(x[i, :])))]

        return output[1:]  # Cut of the first one because that is the empty array input

    def get_name(self) -> str:
        return self.__class__.__name__

    def plot_data(
        self, data: Data, px: int = 300, domain: np.ndarray = np.tile([0.0, 1.0], (2, 1))
    ):  # pragma: no cover
        """Create a 2D contour plot with the datapoints as scatter

        Args:
            data (Data): Data object containing samples
            px (int, optional): Number of pixels on each axis. Defaults to 300.
            domain (np.ndarray, optional): Domain that needs to be plotted. Defaults to np.tile([0.0, 1.0], (2, 1)).

        Returns:
            _type_: _description_
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
        x1_best = data.get_n_best_output_samples(nosamples=1).iloc[:, 0]
        x2_best = data.get_n_best_output_samples(nosamples=1).iloc[:, 1]
        ax.scatter(x=x1_best, y=x2_best, s=25, c="red", marker="*", edgecolors="red")
        return fig, ax

    def plot(
        self,
        orientation: str = "3D",
        px: int = 300,
        domain: np.ndarray = np.tile([0.0, 1.0], (2, 1)),
        show: bool = True,
    ):  # pragma: no cover
        """Generate a surface plot, either 2D or 3D, of the function

        Args:
            orientation (str, optional): Either "2D" or "3D" orientation. Defaults to "3D".
            px (int, optional): Number of points per dimension. Defaults to 300.
            domain (List, optional): Domain that needs to be plotted . Defaults to [0, 1].

        Returns:
            fig, ax: Figure and axis
        """

        if not show:
            plt.ioff()
        else:
            plt.ion()

        x1 = np.linspace(domain[0, 0], domain[0, 1], num=px)
        x2 = np.linspace(domain[1, 0], domain[1, 1], num=px)
        X1, X2 = np.meshgrid(x1, x2)

        Y = np.zeros([len(X1), len(X2)])

        for i in range(len(X1)):
            for j in range(len(X1)):
                xy = np.array([X1[i, j], X2[i, j]] + [0.0] * (self.dimensionality - 2))
                Y[i, j] = self(xy)

        # Add absolute value of global minimum + epsilon to ensure positivity
        # if (
        #     self.get_global_minimum(self.dimensionality)[1][0] < 0
        #     and self.get_global_minimum(self.dimensionality) is not None
        # ):
        #     Y += np.abs(self.get_global_minimum(self.dimensionality)[1][0]) + 10e-6

        dx = (domain[0, 1] - domain[0, 0]) / px
        dy = (domain[1, 1] - domain[1, 0]) / px
        x = domain[0, 0] + dx * np.arange(Y.shape[0])
        y = domain[1, 0] + dy * np.arange(Y.shape[1])
        xv, yv = np.meshgrid(x, y)

        fig = plt.figure(figsize=(7, 7), constrained_layout=True)
        if orientation == "2D":
            ax = plt.axes()
            ax.pcolormesh(xv, yv, Y, cmap="viridis", norm=mcol.LogNorm())  # mcol.LogNorm()
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

    def _scale_input(self, x: np.ndarray) -> np.ndarray:
        return _scale_vector(x=_descale_vector(x, scale=self.scale_bounds), scale=self.input_domain)

    def _descale_input(self, x: np.ndarray) -> np.ndarray:
        return _scale_vector(x=_descale_vector(x, scale=self.input_domain), scale=self.scale_bounds)

    def _from_input_to_scaled(self, x: np.ndarray) -> np.ndarray:

        x = self._reshape_input(x)

        x = self._offset_input(x)
        # x = self._rotate_input(x)

        x = self._scale_input(x)

        return x

    def _retrieve_original_input(self, x: np.ndarray) -> np.ndarray:

        x = self._reshape_input(x)

        x = self._descale_input(x)

        # x = self._reverse_rotate_input(x)
        x = self._reverse_offset_input(x)

        return x

    def _check_global_minimum(self) -> np.ndarray:
        global_minimum_method = getattr(self, "get_global_minimum", None)
        if callable(global_minimum_method):
            g = self.get_global_minimum(d=self.dimensionality)[0]

            if g is None:
                g = np.zeros(self.dimensionality)

            if g.ndim == 2:
                g = g[0]

        else:
            g = np.zeros(self.dimensionality)

        return g

    def _create_offset(self):
        self.offset = np.zeros(self.dimensionality)

        global_minimum_method = getattr(self, "get_global_minimum", None)
        if callable(global_minimum_method):
            g = self.get_global_minimum(d=self.dimensionality)[0]

            if g is None:
                g = np.zeros(self.dimensionality)

            if g.ndim == 2:
                g = g[0]

        else:
            g = np.zeros(self.dimensionality)

        unscaled_offset = np.atleast_2d(
            [
                np.random.uniform(
                    low=-abs(g[d] - self.scale_bounds[d, 0]), high=abs(g[d] - self.scale_bounds[d, 1])
                )  # Here a bug
                for d in range(self.dimensionality)
            ]
        )

        self.offset = unscaled_offset

    def _create_rotation_point(self):
        global_minimum_method = getattr(self, "get_global_minimum", None)
        if callable(global_minimum_method):
            g = self.get_global_minimum(d=self.dimensionality)[0]

            if g is None:
                g = np.zeros(self.dimensionality)

            if g.ndim == 2:
                g = g[0]

        else:
            g = np.zeros(self.dimensionality)

        self.rotation_point = g

    def _create_rotation(self):
        self.rotation_matrix = special_ortho_group(dim=self.dimensionality, seed=self.seed).rvs()

    def _rotate_input(self, x: np.ndarray) -> np.ndarray:
        x = x - self.rotation_point
        x = _rotate_vector(x=x, rotation_matrix=self.rotation_matrix)  # Orthogonal matrix
        x = x + self.rotation_point
        return x

    def _reverse_rotate_input(self, x: np.ndarray) -> np.ndarray:
        x = x + self.rotation_point
        x = _rotate_vector(x=x, rotation_matrix=self.rotation_matrix.T)  # Orthogonal matrix
        x = x - self.rotation_point
        return x

    def _add_noise(self, y: np.ndarray) -> np.ndarray:
        # TODO: change noise calculation to work with autograd.numpy
        """Add Gaussian noise to the output of the function

        Args:
            y (np.ndarray): output of the objective function

        Returns:
            np.ndarray: output of the objective function with added noise
        """
        # sigma = 0.2  # Hard coded amount of noise
        noise: np.ndarray = np.random.normal(loc=0.0, scale=abs(self.noise * y), size=y.shape)
        y_noise = y + noise
        return y_noise

    def _set_parameters(self):
        pass

    def _reshape_input(self, x: np.ndarray) -> np.ndarray:
        # x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
        if x.ndim == 1:
            x = np.reshape(x, (-1, len(x)))  # reshape into 2d array

        return x

    def _offset_input(self, x: np.ndarray) -> np.ndarray:
        return x - self.offset

    def _reverse_offset_input(self, x: np.ndarray) -> np.ndarray:
        return x + self.offset
