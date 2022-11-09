"""
Information on the Adamax optimizer
"""

from dataclasses import dataclass

import tensorflow as tf

from ..base.optimization import OptimizerParameters
from .adapters.tensorflow_implementations import TensorflowOptimizer


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
