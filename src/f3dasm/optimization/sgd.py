from dataclasses import dataclass

import tensorflow as tf

from ..base.optimization import OptimizerParameters
from .adapters.tensorflow_implementations import TensorflowOptimizer


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
