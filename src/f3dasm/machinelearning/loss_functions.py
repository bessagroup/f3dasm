#                                                                       Modules
# =============================================================================

# Local
from .._imports import try_import

# Third-party extension
with try_import('machinelearning') as _imports:
    import tensorflow as tf

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================


def MeanSquaredError(Y_pred, Y_true):
    """Mean squared error (MSE) loss function

    Parameters
    ----------
    Y_pred
        Predicted labels
    Y_true
        True labels

    Returns
    -------
        Float value denoting the mean squared error of the model
    """
    fn = tf.keras.losses.MeanSquaredError()
    return fn(Y_true, Y_pred)
