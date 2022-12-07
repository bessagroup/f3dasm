#                                                                       Modules
# =============================================================================

# Standard


# Third-party
import autograd.numpy as np

# Locals
from ..base.function import Function

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Alpha'
# =============================================================================
#
# =============================================================================


class LogisticRegression(Function):  # 4D, minimal 2D!
    def _set_parameters(self):
        _means = np.random.random(size=(2, self.dimensionality))
        _cov = np.random.random((self.dimensionality, self.dimensionality, 2))
        _labels = np.array([0, 1])

        self._xi, self._yi = [], []
        xx1 = np.random.multivariate_normal(
            _means[0], np.dot(_cov[:, :, 0], _cov[:, :, 0].T), size=50)
        xx2 = np.random.multivariate_normal(
            _means[1], np.dot(_cov[:, :, 1], _cov[:, :, 1].T), size=50)

        self._xi = np.r_[xx1, xx2]
        self._yi = np.array([0] * 50 + [1] * 50)

        orig_bounds = [-5.0, 5.0]

    def _s_func(self, z):
        return 1 / (1 + np.exp(-z))

    def f(self, x: np.ndarray):
        l_term = 0.0005
        ww = x[:-1]
        bb = x[-1]
        c = (
            -1
            / 100
            * sum(
                self._yi * np.log10(self._s_func(np.inner(ww, self._xi) + bb))
                + (1 - self._yi) * np.log10(1 -
                                            self._s_func(np.inner(ww, self._xi) + bb))
                + l_term / 2 * (np.sqrt(ww.dot(ww))) ** 2
            )
        )

        if np.isinf(c) or np.isnan(c):
            c = 10e5

        return c
