#                                                                       Modules
# =============================================================================

# Standard
from functools import partial
from typing import Any

# Third-party
import tensorflow as tf

# Local
from .adapters.tensorflow_implementations import TensorflowModel

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class PassthroughLayer(tf.keras.layers.Layer):
    def __init__(self, input_shape, units=1):
        super().__init__(input_shape=input_shape)
        self.units = units

    def build(self, input_shape):  # Create the state of the layer (weights)
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer='random_normal', trainable=True)
        # self.w = self.add_weight(shape=(self.units, input_shape[-1]), initializer='random_normal', trainable=True)
    def call(self, inputs):  # Defines the computation from inputs to outputs
        # Whatever you put through the model, it will just output the weights!
        # But transposed!
        w_transpose = tf.transpose(self.w)
        return w_transpose  # self.w


class PassthroughModel(TensorflowModel):
    def __init__(self, dimensionality: int):  # loss_function
        super().__init__()
        self.model.add(PassthroughLayer(input_shape=(dimensionality,)))

        # # Loss function is a benchmark function
        # self._loss_function = loss_function

        # # We don't have labels for benchmark function loss
        # self.loss = partial(self.loss, Y_true=None)
