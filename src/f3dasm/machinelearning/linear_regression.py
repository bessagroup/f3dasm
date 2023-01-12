#                                                                       Modules
# =============================================================================

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


class LinearRegression(TensorflowModel):
    def __init__(self, dimensionality: int):  # introduce a dimensionality parameter because trainable weights!
        super().__init__()
        self.model.add(tf.keras.layers.Dense(1, input_shape=(dimensionality,)))
