#                                                                       Modules
# =============================================================================

# Standard
from dataclasses import dataclass
from typing import List

# Locals
from .._imports import try_import
from .adapters.tensorflow_implementations import TensorflowOptimizer
from .optimizer import OptimizerParameters

# Third-party extension
with try_import('optimization') as _imports:
    import tensorflow as tf

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


@dataclass
class RMSprop_Parameters(OptimizerParameters):
    """Hyperparameters for RMSprop optimizer"""

    learning_rate: float = 0.001
    rho: float = 0.9
    momentum: float = 0.0
    epsilon: float = 1e-07
    centered: bool = False


class RMSprop(TensorflowOptimizer):
    """RMSprop"""

    parameter: RMSprop_Parameters = RMSprop_Parameters()

    def set_algorithm(self):
        self.algorithm = tf.keras.optimizers.RMSprop(
            learning_rate=self.parameter.learning_rate,
            rho=self.parameter.rho,
            momentum=self.parameter.momentum,
            epsilon=self.parameter.epsilon,
            centered=self.parameter.centered,
        )

    def get_info(self) -> List[str]:
        return ['Stable', 'Single-Solution']
