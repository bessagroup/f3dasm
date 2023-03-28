#                                                                       Modules
# =============================================================================

# Local
from .._imports import try_import

# Third-party extension
with try_import('machinelearning') as _imports:
    import tensorflow as tf
    from keras.layers import Layer

    from .adapters.tensorflow_implementations import TensorflowModel

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

if not _imports.is_successful():
    Layer = object  # NOQA
    TensorflowModel = object # NOQA


class _PassthroughLayer(Layer):
    def __init__(self, input_shape, units=1):
        super().__init__(input_shape=input_shape)
        self.units = units

    def build(self, input_shape):
        """Create the state of the layer (weights)

        Parameters
        ----------
        input_shape
            input shape of the layer
        """
        self.w = self.add_weight(shape=(input_shape[-1], self.units), initializer='random_normal',
                                 trainable=True, dtype=tf.float64)

    def call(self, inputs):
        """Defines the computation from inputs to outputs

        Parameters
        ----------
        inputs
            input of the layer
        """
        # Whatever you put through the model, it will just output the weights, but transposed!
        w_transpose = tf.transpose(self.w)
        return w_transpose


class PassthroughModel(TensorflowModel):
    def __init__(self, dimensionality: int):  # loss_function
        """Model that passed through the input layer directly to the output layer

        Parameters
        ----------
        dimensionality
            number of input parameters
        """
        _imports.check()
        self.dimensionality = dimensionality
        super().__init__()
        self.model.add(_PassthroughLayer(input_shape=(dimensionality,)))

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'dimensionality': self.dimensionality, })
        return config
