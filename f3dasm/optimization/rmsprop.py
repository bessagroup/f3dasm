from dataclasses import dataclass

from ..base.function import Function
from ..base.optimization import Optimizer, OptimizerParameters

from .adapters.tensorflow_implementations import TensorflowOptimizer
import tensorflow as tf


@dataclass
class RMSprop_Parameters(OptimizerParameters):
    """Hyperparameters for Momentum optimizer"""

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
