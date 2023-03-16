#                                                                       Modules
# =============================================================================

# Standard
from typing import List
from unittest import mock

# Third-party core
import numpy as np

# Local
from ..._imports import try_import
from ..model import Model

# Third-party extension
with try_import('machinelearning') as _imports:
    import tensorflow as tf
    from keras import Model as tf_Model

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

if not _imports.is_successful():
    tf_Model = object  # NOQA


class TensorflowModel(tf_Model, Model):

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



def get_reshaped_array_from_list_of_arrays(flat_array: np.ndarray,
                                           list_of_arrays: List[np.ndarray]) -> List[np.ndarray]:
    total_array = []
    index = 0
    for mimic_array in list_of_arrays:
        number_of_values = np.product(mimic_array.shape)
        current_array = np.array(flat_array[index:index+number_of_values])

        if number_of_values > 1:
            current_array = current_array.reshape(-1, 1)  # Make 2D array

        total_array.append(current_array)
        index += number_of_values

    return total_array


# technically not a np array input!
def get_flat_array_from_list_of_arrays(list_of_arrays: List[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.atleast_2d(array) for array in list_of_arrays])
