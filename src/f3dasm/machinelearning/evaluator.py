#                                                                       Modules
# =============================================================================

# Standard
from typing import Callable, Tuple, Union

# Third-party core
import autograd.numpy as np

# Locals
from .._imports import try_import
from ..data.learningdata import LearningData
from .adapters.tensorflow_implementations import \
    get_flat_array_from_list_of_arrays
from .loss_functions import MeanSquaredError
from .model import Model
from .passthrough_model import PassthroughModel

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


class Evaluator():
    def __init__(self, loss_function: Callable = MeanSquaredError, model: Union[Model, None] = None,
                 learning_data: Union[LearningData, None] = None):
        """Combines a loss function, machine learning model and learning data to evaluate a model

        Parameters
        ----------
        loss_function, optional
            Loss function to evaluate the loss of predicted labels and true labels, by default MeanSquaredError
        model, optional
            Machine learning model, by default None
        learning_data, optional
            Data to go through the model to calculate the predicted labesl, by default None
        """
        _imports.check()
        self.loss_function = loss_function
        self.model = model
        self.learning_data = learning_data

    # Two 2D arrays: loss (1,1), grad (dim, 1)
    def evaluate(self, weights: Union[np.ndarray, None] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate the evaluator to calculate the loss and gradients

        Parameters
        ----------
        x, optional
            weights of the network to evaluate, by default None

        Returns
        -------
            Tuple of loss and gradients
        """

        # If there is no learning data
        if self.learning_data is None:
            X_data = weights  # or None !
            y_data = None
        else:
            X_data = self.learning_data.get_input_data()
            y_data = self.learning_data.get_labels()

        # If there is no model
        if self.model is None:
            # Create the passthroughmodel based on the dimensionality
            self.model = PassthroughModel(dimensionality=weights.size)  # product of x.shape

        self.model.set_model_weights(weights)

        with tf.GradientTape() as tape:
            loss = self.loss_function(self.model(X_data), Y_true=y_data)
        grads = tape.gradient(loss, self.model.trainable_variables)  # = dependent on tensorflow !!

        dydx = get_flat_array_from_list_of_arrays(grads)
        y = np.atleast_2d(loss.numpy())

        return y, dydx

    def tf_apply_gradients(self, weights: np.ndarray, optimizer: tf.keras.optimizers.Optimizer):
        # If there is no learning data
        if self.learning_data is None:
            X_data = weights  # or None !
            y_data = None
        else:
            X_data = self.learning_data.get_input_data()
            y_data = self.learning_data.get_labels()

        # If there is no model
        if self.model is None:
            # Create the passthroughmodel based on the dimensionality
            self.model = PassthroughModel(dimensionality=weights.size)  # product of x.shape

        self.model.set_model_weights(weights)

        with tf.GradientTape() as tape:
            loss = self.loss_function(self.model(X_data), Y_true=y_data)
        grads = tape.gradient(loss, self.model.trainable_variables)  # = dependent on tensorflow !!

        optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        x = self.model.get_model_weights().T
        y = self(x)
        return x, y

    def f(self, x: np.ndarray):
        x = np.atleast_2d(x)
        y = []
        for xi in x:
            yi, _ = self.evaluate(xi)
            y.append(yi)

        loss = np.array(y).reshape(-1, 1)

        # loss, _ = self.evaluate(x)
        return loss

    def __call__(self, x: np.ndarray):
        return self.f(x)

    def dfdx(self, x: np.ndarray):
        x = np.atleast_2d(x)
        dfdx = []
        for xi in x:
            _, dfdxi = self.evaluate(xi)
            dfdx.append(dfdxi)

        grads = np.array(dfdx).reshape(-1, 1)

        # _, grads = self.evaluate(x)
        return grads

    def dfdx_legacy(self, x: np.ndarray):
        return self.dfdx(x)
