#                                                                       Modules
# =============================================================================

# Standard
from copy import copy
from typing import Any

# Third-party
import autograd.numpy as np

# Locals
from ...base.function import Function
from ...base.utils import _descale_vector, _scale_vector
from .augmentor import FunctionAugmentor, Noise, Offset, Scale

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class PyBenchFunction(Function):
    def __init__(
        self,
        dimensionality: int = 2,
        noise: Any or float = None,
        seed: Any or int = None,
        scale_bounds: Any or np.ndarray = None,
    ):
        """Adapter for pybenchfunctions, created by Axel Thevenot (2020).
        Github repository: https://github.com/AxelThevenot/Python_Benchmark_Test_Optimization_Function_Single_Objective


        :param dimensionality: number of dimensions
        :param noise: inflict Gaussian noise on the input
        :param seed: seed for the random number generator
        :param scale_bounds: array containing the lower and upper bound of the scaling factor of the input data
        """
        self.noise = noise
        self.offset = np.zeros(dimensionality)
        self.input_domain: Any or np.ndarray = None
        self.augmentor = FunctionAugmentor()

        super().__init__(dimensionality=dimensionality, seed=seed)

        self._create_scale_bounds(scale_bounds)

        self._create_offset()

        # TEMP
        # self.offset = np.zeros(dimensionality)

        self.augmentor = self._construct_augmentor()

    @classmethod
    def is_dim_compatible(cls, d: int) -> bool:
        """Check if the functdion is compatible with a certain number of dimenions

        :param d: number of dimensions
        :return:
        """
        pass

    def _construct_augmentor(self) -> FunctionAugmentor:

        input_augmentors = [
            Offset(offset=self.offset),
            Scale(scale_bounds=self.scale_bounds,
                  input_domain=self.input_domain),
        ]
        output_augmentors = []

        if self.noise not in [None, "None"]:
            output_augmentors.append(Noise(noise=self.noise))

        return FunctionAugmentor(input_augmentors=input_augmentors, output_augmentors=output_augmentors)

    def _create_scale_bounds(self, input: Any):
        if input is None:
            self.scale_bounds = np.tile([0.0, 1.0], (self.dimensionality, 1))
        else:
            self.scale_bounds = input

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

        unscaled_offset = np.atleast_1d(
            [
                np.random.uniform(
                    low=-abs(g[d] - self.scale_bounds[d, 0]), high=abs(g[d] - self.scale_bounds[d, 1]))
                for d in range(self.dimensionality)
            ]
        )

        self.offset = unscaled_offset

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

    def _add_noise(self, y: np.ndarray) -> np.ndarray:
        # TODO: change noise calculation to work with autograd.numpy
        # sigma = 0.2  # Hard coded amount of noise
        if hasattr(y, "_value"):
            yy = copy(y._value)
            if hasattr(yy, "_value"):
                yy = copy(yy._value)
        else:
            yy = copy(y)

        scale = abs(self.noise * yy)

        noise: np.ndarray = np.random.normal(
            loc=0.0, scale=scale, size=y.shape)
        y_noise = y + float(noise)
        return y_noise

    def check_if_within_bounds(self, x: np.ndarray) -> bool:
        """Check if the input vector is between the given scaling bounds

        :param x: input vector
        :return: wheter the vector is within the boundaries
        """
        return ((self.scale_bounds[:, 0] <= x) & (x <= self.scale_bounds[:, 1])).all()

    def _scale_input(self, x: np.ndarray) -> np.ndarray:
        return _scale_vector(x=_descale_vector(x, scale=self.scale_bounds), scale=self.input_domain)

    def _descale_input(self, x: np.ndarray) -> np.ndarray:
        return _scale_vector(x=_descale_vector(x, scale=self.input_domain), scale=self.scale_bounds)

    def _offset_input(self, x: np.ndarray) -> np.ndarray:
        return x - self.offset

    def _reverse_offset_input(self, x: np.ndarray) -> np.ndarray:
        return x + self.offset

    def _retrieve_original_input(self, x: np.ndarray) -> np.ndarray:

        x = self._reshape_input(x)

        # s = Scale(scale_bounds=self.scale_bounds, input_domain=self.input_domain)
        # x = s.reverse_augment(x)
        x = self._descale_input(x)

        # o = Offset(offset=self.offset)
        # x = o.reverse_augment(x)
        x = self._reverse_offset_input(x)

        # x_out = self.augmentor.augment_reverse_input(x)
        return x

    def evaluate(x: np.ndarray):
        """Evaluate the objective function

        :param x: input fector
        :raises NotImplementedError: If no function is implemented
        """
        raise NotImplementedError("No function implemented!")

    def f(self, x: np.ndarray):
        """Analytical form of the objective function

        :param x: input vector
        :return: objective value
        """
        if self.is_dim_compatible(self.dimensionality):
            y = []
            for xi in x:

                # o = Offset(offset=self.offset)
                # xi = o.augment(xi)
                xi = self._offset_input(xi)

                # s = Scale(scale_bounds=self.scale_bounds, input_domain=self.input_domain)
                # xi = s.augment(xi)
                xi = self._scale_input(xi)

                # xi = self.augmentor.augment_input(xi)

                yi = self.evaluate(xi)

                # add noise
                if self.noise not in [None, "None"]:
                    yi = self._add_noise(yi)

                y.append(yi)

        else:
            raise ValueError("Dimension is not compatible with function!")

        return np.array(y).reshape(-1, 1)
