#                                                                       Modules
# =============================================================================

# Standard
from typing import Any, Tuple

# Third-party
import autograd.numpy as np
import matplotlib.colors as mcol
import matplotlib.pyplot as plt
import tensorflow as tf

# Locals
from ..base.data import Data
from ..base.utils import (SimpelModel,
                          _from_data_to_numpy_array_benchmarkfunction,
                          convert_autograd_to_tensorflow)

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class Function:
    def __init__(self, dimensionality: int, seed: Any or int = None):
        """Interface of a continuous benchmark function

        Parameters
        ----------
        dimensionality :
            number of input dimensions
        seed : Any, optional
            value to seed the random generator, by default None
        """
        self.dimensionality = dimensionality
        self.seed = seed
        self.__post_init__()

    def __post_init__(self):

        if self.seed:
            self.set_seed(self.seed)

        self._set_parameters()
        self._tf_gradient_setup()

    def _tf_gradient_setup(self):
        self.args = {}

        self.args["model"] = SimpelModel(
            None,
            args={
                "dim": self.dimensionality,
                "x0": np.zeros(self.dimensionality),
            },
        )  # Build the model
        self.args["tvars"] = self.args["model"].trainable_variables
        self.args["func"] = convert_autograd_to_tensorflow(self.__call__)

    def set_seed(self, seed: int):
        """Set the numpy seed of the random generator

        Parameters
        ----------
        seed
            seed for random number generator
        """
        self.seed = seed
        np.random.seed(seed)

    def __call__(self, input_x: np.ndarray or Data, *args, **kwargs) -> np.ndarray or Data:
        """Evaluate the objective function

        Parameters
        ----------
        input_x
            input to be evaluated

        Returns
        -------
            output of the objective function
        """
        # If the input is a Data object
        if isinstance(input_x, Data):
            x = _from_data_to_numpy_array_benchmarkfunction(data=input_x)

        else:
            x = input_x

        x = self._reshape_input(x)

        y = np.atleast_1d(self.f(x))

        # If the input is a Data object
        if isinstance(input_x, Data):
            input_x.add_output(y)
            # return input_x

        return y

    def f(self, x) -> np.ndarray:
        """Analytical function of the objective function. Needs to be implemented by inhereted class

        Parameters
        ----------
        x
            input vector

        Returns
        -------
            output vector

        Raises
        ------
        NotImplementedError
            Raised when not implemented. Subclasses should implement this method.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def dfdx(self, x: np.ndarray) -> np.ndarray:
        # Need to ravel x to be shape = (dim,)
        self.args["model"].z.assign(x)

        with tf.GradientTape() as tape:
            tape.watch(self.args["tvars"])
            logits = 0.0 + self.args["model"](None)  # tf.cast(self.args["model"](None), tf.float64)
            loss = self.args["func"](tf.reshape(
                logits, (self.dimensionality)))

        grads = tape.gradient(loss, self.args["tvars"])
        return grads[0].numpy().copy()

    def dfdx_legacy(self, x: np.ndarray, dx=1e-8) -> np.ndarray:
        """Compute the gradient at a particular point in space. Gradient is computed by numdifftools

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

    def get_name(self) -> str:
        """Get the name of the function

        Returns
        -------
            name of the function
        """
        return self.__class__.__name__

    def plot_data(
        self, data: Data, px: int = 300, domain: np.ndarray = np.array([[0.0, 1.0], [0.0, 1.0]]),
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

    def _set_parameters(self):
        pass

    def _reshape_input(self, x: np.ndarray) -> np.ndarray:
        # x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf, doenst work with autograd.numpy
        try:
            if x.ndim == 1:
                x = np.reshape(x, (-1, len(x)))  # reshape into 2d array
        except AttributeError:
            return x

        return x
