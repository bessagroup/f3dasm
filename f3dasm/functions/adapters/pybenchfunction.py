from dataclasses import field
from typing import Any

import autograd.numpy as np

from ...base.function import Function
from ...base.utils import _descale_vector, _scale_vector


class PyBenchFunction(Function):
    """
    Adapter for pybenchfunctions, created by Axel Thevenot (2020)
    Github repository: https://github.com/AxelThevenot/Python_Benchmark_Test_Optimization_Function_Single_Objective

    scale_bounds (Any|np.ndarray): array containing the lower and upper bound of the scaling factor of the input data (Default = [0.0, 1.0])
    input_domain (Any|np.ndarray): array containing the lower and upper bound of the input domain of the original function (Default = [0.0, 1.0])
    """

    def __init__(
        self,
        dimensionality: int = 2,
        noise: Any or float = None,
        seed: Any or int = None,
        scale_bounds: Any or np.ndarray = None,
    ):
        self.input_domain: Any or np.ndarray = None
        super().__init__(dimensionality=dimensionality, noise=noise, seed=seed)
        self._create_scale_bounds(scale_bounds)

        self._create_offset()

    @classmethod
    def is_dim_compatible(cls, d) -> bool:
        pass

    def _create_scale_bounds(self, input: Any) -> None:
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

        unscaled_offset = np.atleast_2d(
            [
                np.random.uniform(low=-abs(g[d] - self.scale_bounds[d, 0]), high=abs(g[d] - self.scale_bounds[d, 1]))
                for d in range(self.dimensionality)
            ]
        )

        self.offset = unscaled_offset

    def check_if_within_bounds(self, x: np.ndarray) -> bool:
        """Check if the input vector is between the given scaling bounds

        Args:
            x (np.ndarray): input vector

        Returns:
            bool: whether the vector is within the boundaries
        """
        return ((self.scale_bounds[:, 0] <= x) & (x <= self.scale_bounds[:, 1])).all()

    def _scale_input(self, x: np.ndarray) -> np.ndarray:
        return _scale_vector(x=_descale_vector(x, scale=self.scale_bounds), scale=self.input_domain)

    def _descale_input(self, x: np.ndarray) -> np.ndarray:
        return _scale_vector(x=_descale_vector(x, scale=self.input_domain), scale=self.scale_bounds)

    def _retrieve_original_input(self, x: np.ndarray) -> np.ndarray:

        x = self._reshape_input(x)

        x = self._descale_input(x)

        # x = self._reverse_rotate_input(x)
        x = self._reverse_offset_input(x)

        return x

    def evaluate(x: np.ndarray):
        raise NotImplementedError("No function implemented!")

    def f(self, x: np.ndarray):
        if self.is_dim_compatible(self.dimensionality):
            y = []
            x = np.atleast_2d(x)
            for xi in x:

                xi = self._scale_input(xi)

                y.append(self.evaluate(xi))

            return np.array(y).reshape(-1, 1)
