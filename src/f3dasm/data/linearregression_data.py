#                                                                       Modules
# =============================================================================

# Standard
from typing import List

# Third-party
import numpy as np

# Locals
from .learningdata import LearningData

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class LinearRegressionData(LearningData):
    def __init__(self, n: int, b: float, w: List[float], noise_multiplier: float = 0.01):
        """Class to store and create randomly sampled regression data

        Parameters
        ----------
        n
            number of samples
        b
            value of the offset
        w
            value of the weights
        noise_multiplier, optional
            standard deviation of the noise, by default 0.01
        """
        self.n = n
        self.b = b
        self.w = w
        self.noise_multiplier = noise_multiplier
        self._create()

    def _create(self):
        w = np.array(self.w, dtype=float)
        noise = np.random.normal(size=(self.n, 1)) * self.noise_multiplier

        self.X = np.random.normal(size=(self.n, w.shape[0]))
        self.y = np.matmul(self.X, np.reshape(w, (-1, 1))) + self.b + noise

    def get_input_data(self) -> np.ndarray:  # size = (n, dim)
        return self.X

    def get_labels(self) -> np.ndarray:  # size = (n, 1)
        return self.y
