#                                                                       Modules
# =============================================================================


# Standard
from typing import List

# Third-party
import numpy as np
import tensorflow as tf

# Local
from ..models import Model
from ...base.utils import (get_flat_array_from_list_of_arrays,
                           get_reshaped_array_from_list_of_arrays)

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
