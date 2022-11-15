from dataclasses import field
from typing import Any

import autograd.numpy as np

from ...base.function import Function
from ...base.utils import _descale_vector, _scale_vector


class PyBenchFunction(Function):
    """
    Adapter for pybenchfunctions, created by Axel Thevenot (2020)
    Github repository: https://github.com/AxelThevenot/Python_Benchmark_Test_Optimization_Function_Single_Objective
    """

    input_domain: Any or np.ndarray = field(init=False)

    @classmethod
    def is_dim_compatible(cls, d) -> bool:
        pass

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
            # TODO: instead of editing in place, return new arrays
            # a is a 1-D slice of arr along axis.
            # TODO: create expression for a
            # axis = 1
            # for a in x:
            #     Ni, Nk = a.shape[:axis], a.shape[axis+1:]
            #     for ii in np.ndindex(Ni):
            #         for kk in np.ndindex(Nk):
            #             f = self.evaluate(x[ii + np.s_[:,] + kk])
            #             Nj = f.shape
            #             for jj in np.ndindex(Nj):
            #                 out[ii + jj + kk] = f[jj]
            y = []
            x = np.atleast_2d(x)
            for xi in x:

                xi = self._scale_input(xi)

                y.append(self.evaluate(xi))

            return np.array(y).reshape(-1, 1)

            # return np.apply_along_axis(self.evaluate, axis=1, arr=x)  # .reshape(-1, 1)
