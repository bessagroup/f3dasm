#                                                                       Modules
# =============================================================================

# Standard
from typing import List, Protocol, Tuple

# Third-party
import numpy as np
import tensorflow as tf

# Locals
from ..base.data import Data
from ..base.utils import get_flat_array_from_list_of_arrays

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


class Evaluator():  # Dit moet eigenlijk een soort Function worden, maar dan met een ML architectuur en Data ...
    def __init__(self, model: Model = None, data: Data = None, loss_function=None):
        self.model = model
        self.data = data
        self.loss_function = loss_function

        # self.dimensionality = data.design.get_number_of_input_parameters()

    def evaluate(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:  # Two 2D arrays: loss (1,1), grad (dim, 1)

        self.model.set_model_weights(x)

        if self.data is None:
            X_data = x
            y_data = None
        else:
            X_data = self.data.get_input_data().to_numpy()
            y_data = self.data.get_output_data().to_numpy()

        with tf.GradientTape() as tape:
            loss = self.loss_function(Y_pred=self.model(X_data), Y_true=y_data)
            # loss = self.model.loss(Y_pred=self.model(X_data), Y_true=y_data)
        grads = tape.gradient(loss, self.model.trainable_variables)  # = dependent on tensorflow !!
        return np.atleast_2d(loss.numpy()), get_flat_array_from_list_of_arrays(grads)

    def f(self, x: np.ndarray):
        loss, _ = self.evaluate(x)
        return loss

    def __call__(self, x: np.ndarray):
        return self.f(x)

    def dfdx(self, x: np.ndarray):
        _, grads = self.evaluate(x)
        return grads
