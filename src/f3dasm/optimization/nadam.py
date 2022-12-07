#                                                                       Modules
# =============================================================================

# Standard
from dataclasses import dataclass

# Third-party
import tensorflow as tf

# Locals
from ..base.optimization import OptimizerParameters
from .adapters.tensorflow_implementations import TensorflowOptimizer

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


@dataclass
class Nadam_Parameters(OptimizerParameters):
    """Hyperparameters for Momentum optimizer
    )
    """

    learning_rate: float = 0.001
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-07


class Nadam(TensorflowOptimizer):
    """Nadam"""

    parameter: Nadam_Parameters = Nadam_Parameters()

    def set_algorithm(self):
        self.algorithm = tf.keras.optimizers.Nadam(
            learning_rate=self.parameter.learning_rate,
            beta_1=self.parameter.beta_1,
            beta_2=self.parameter.beta_2,
            epsilon=self.parameter.epsilon,
        )
