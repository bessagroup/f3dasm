#                                                                       Modules
# =============================================================================

# Local
from .._imports import try_import

# Third-party extension
with try_import('machinelearning') as _imports:
    import tensorflow
    import tensorflow as tf

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
    TensorflowModel = object  # NOQA


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
