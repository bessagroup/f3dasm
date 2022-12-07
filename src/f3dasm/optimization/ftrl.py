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
class Ftrl_Parameters(OptimizerParameters):
    """Hyperparameters for Ftrl optimizer"""

    learning_rate: float = 0.001
    learning_rate_power: float = -0.5
    initial_accumulator_value: float = 0.1
    l1_regularization_strength: float = 0.0
    l2_regularization_strength: float = 0.0
    l2_shrinkage_regularization_strength: float = 0.0
    beta: float = 0.0


class Ftrl(TensorflowOptimizer):
    """Ftrl"""

    parameter: Ftrl_Parameters = Ftrl_Parameters()

    def set_algorithm(self):
        self.algorithm = tf.keras.optimizers.Ftrl(
            learning_rate=self.parameter.learning_rate,
            learning_rate_power=self.parameter.learning_rate_power,
            initial_accumulator_value=self.parameter.initial_accumulator_value,
            l1_regularization_strength=self.parameter.l1_regularization_strength,
            l2_regularization_strength=self.parameter.l2_regularization_strength,
            l2_shrinkage_regularization_strength=self.parameter.l2_shrinkage_regularization_strength,
            beta=self.parameter.beta,
        )
