import numpy as np


class Function:
    def eval(self, x, noise=False):
        x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
        if x.ndim == 1:
            x = np.reshape(x, (-1, len(x)))  # reshape into 2d array

        return np.atleast_1d(self.f(x))
