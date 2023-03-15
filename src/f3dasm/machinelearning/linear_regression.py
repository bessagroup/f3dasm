#                                                                       Modules
# =============================================================================

# Local
from .._imports import try_import
from .adapters.tensorflow_implementations import TensorflowModel

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


class LinearRegression(TensorflowModel):
    def __init__(self, dimensionality: int):
        """Linear Regression model

        Parameters
        ----------
        dimensionality
            number of neurons in the first layer
        """
        _imports.check()
        self.dimensionality = dimensionality
        super().__init__()
        self.model.add(tf.keras.layers.Dense(1, input_shape=(dimensionality,)))

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'dimensionality': self.dimensionality, })
        return config
