#                                                                       Modules
# =============================================================================

# Standard
from typing import List, Protocol, Tuple

# Third-party
import numpy as np
import tensorflow as tf

from ..base.utils import get_flat_array_from_list_of_arrays
# Locals
from ..design.experimentdata import ExperimentData

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling', 'https://d2l.ai/']
__status__ = 'Alpha'
# =============================================================================
#
# =============================================================================


class Model(Protocol):
    """Base class for all machine learning models"""

    def forward(self, X):
        """Forward pass of the model: calculate an output by giving it an input

        Parameters
        ----------
        X
            Input of the model
        """
        # The model should handle the case when X is None!
        ...

    def get_model_weights(self):
        """Retrieve the model weights as a 1D array"""
        ...

    def set_model_weights(self):
        """Set the model weights with a 1D array"""
        ...


def MeanSquaredError(Y_pred, Y_true):
    fn = tf.keras.losses.MeanSquaredError()
    return fn(Y_true, Y_pred)
