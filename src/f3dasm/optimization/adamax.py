"""
Information on the Adamax optimizer
"""
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
class Adamax_Parameters(OptimizerParameters):
    """Hyperparameters for Adamax optimizer"""

    learning_rate: float = 0.001
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-07


class Adamax(TensorflowOptimizer):
    """Adamax"""

    parameter: Adamax_Parameters = Adamax_Parameters()

    def set_algorithm(self):
        self.algorithm = tf.keras.optimizers.Adamax(
            learning_rate=self.parameter.learning_rate,
            beta_1=self.parameter.beta_1,
            beta_2=self.parameter.beta_2,
            epsilon=self.parameter.epsilon,
        )

    def get_info(self) -> List[str]:
        return ['Fast', 'Single-Solution']
