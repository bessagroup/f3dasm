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
class SGD_Parameters(OptimizerParameters):
    """Hyperparameters for Momentum optimizer"""

    learning_rate: float = 0.01
    momentum: float = 0.0
    nesterov: bool = False


class SGD(TensorflowOptimizer):
    """SGD"""

    parameter: SGD_Parameters = SGD_Parameters()

    def set_algorithm(self):
        self.algorithm = tf.keras.optimizers.SGD(
            learning_rate=self.parameter.learning_rate,
            momentum=self.parameter.momentum,
            nesterov=self.parameter.nesterov,
        )

    def get_info(self) -> List[str]:
        return ['Stable', 'First-Order', 'Single-Solution']
