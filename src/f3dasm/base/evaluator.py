#                                                                       Modules
# =============================================================================

# Standard
from typing import Any, Callable, Tuple, Union

# Third-party
import autograd.numpy as np
import tensorflow as tf

# Locals
from ..machinelearning.model import Model
from ..machinelearning.passthrough_model import PassthroughModel
from .learningdata import LearningData
from .utils import get_flat_array_from_list_of_arrays

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


def MeanSquaredError(Y_pred, Y_true):
    fn = tf.keras.losses.MeanSquaredError()
    return fn(Y_true, Y_pred)


class Evaluator():  # Dit moet eigenlijk een soort Function worden, maar dan met een ML architectuur en Data ...
    def __init__(self, loss_function: Callable = MeanSquaredError, model: Union[Model, None] = None,
                 learning_data: Union[LearningData, None] = None):
        self.loss_function = loss_function
        self.model = model
        self.learning_data = learning_data

    # Two 2D arrays: loss (1,1), grad (dim, 1)
    def evaluate(self, x: Union[np.ndarray, None] = None) -> Tuple[np.ndarray, np.ndarray]:

        # If there is no learning data
        if self.learning_data is None:
            X_data = x
            y_data = None
        else:
            X_data = self.learning_data.get_input_data().to_numpy()
            y_data = self.learning_data.get_output_data().to_numpy()

        # If there is no model
        if self.model is None:
            # Create the passthroughmodel based on the dimensionality
            self.model = PassthroughModel(dimensionality=x.size)  # product of x.shape

        self.model.set_model_weights(x)

        with tf.GradientTape() as tape:
            loss = self.loss_function(self.model(X_data), Y_true=y_data)
            # loss = self.loss_function(Y_pred=self.model(X_data), Y_true=y_data)
            # loss = self.loss_function(Y_pred=Y_pred, Y_true=y_data)
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
