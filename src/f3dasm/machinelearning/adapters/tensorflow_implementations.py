#                                                                       Modules
# =============================================================================

# Standard
from typing import List

# Third-party core
import numpy as np

# Local
from ..._imports import try_import
from ...base.utils import (get_flat_array_from_list_of_arrays,
                           get_reshaped_array_from_list_of_arrays)
from ..model import Model

# Third-party extension
with try_import('machinelearning') as _imports:
    import tensorflow as tf

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class TensorflowModel(tf.keras.Model, Model):
    def __init__(self):
        _imports.check()
        super().__init__()
        self.model = tf.keras.models.Sequential()

    def forward(self, X):
        assert hasattr(self, 'model'), 'model is defined'
        return self.model(X)

    def call(self, X, *args, **kwargs):  # Shape: (samples, dim)
        return self.forward(X, *args)

    def get_model_weights(self) -> List[np.ndarray]:
        return get_flat_array_from_list_of_arrays(self.model.get_weights())
        # return self.model.get_weights()

    def set_model_weights(self, weights: np.ndarray):
        reshaped_weights = get_reshaped_array_from_list_of_arrays(
            flat_array=weights.ravel(), list_of_arrays=self.model.get_weights())
        self.model.set_weights(reshaped_weights)


def MeanSquaredError(Y_pred, Y_true):
    """Mean squared error (MSE) loss function

    Parameters
    ----------
    Y_pred
        Predicted labels
    Y_true
        True labels

    Returns
    -------
        Float value denoting the mean squared error of the model
    """
    fn = tf.keras.losses.MeanSquaredError()
    return fn(Y_true, Y_pred)
