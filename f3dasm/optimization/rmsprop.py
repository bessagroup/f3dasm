from dataclasses import dataclass

import tensorflow as tf

from ..base.optimization import OptimizerParameters
from .adapters.tensorflow_implementations import TensorflowOptimizer


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
