#                                                                       Modules
# =============================================================================

# Standard
from typing import List

# Third-party
import numpy as np
import tensorflow as tf

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
        w: tf.Tensor = tf.constant(self.w, dtype=float)
        dim = len(self.w)
        num_train = self.n // 2
        num_val = self.n - num_train

        noise = tf.random.normal((self.n, 1)) * self.noise_multiplier

        self.X: tf.Tensor = tf.random.normal((self.n, w.shape[0]))  # (num, dim)
        self.y: tf.Tensor = tf.matmul(self.X, tf.reshape(w, (-1, 1))) + self.b + noise  # (1, dim)

    def get_input_data(self) -> np.ndarray:  # size = (n, dim)
        return self.X.numpy()

    def get_labels(self) -> np.ndarray:  # size = (n, 1)
        return self.y.numpy()
