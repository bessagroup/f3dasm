from dataclasses import dataclass

from ..base.optimization import OptimizerParameters

from .adapters.tensorflow_implementations import TensorflowOptimizer
import tensorflow as tf


@dataclass
class Adam2_Parameters(OptimizerParameters):
    """Hyperparameters for Momentum optimizer"""

    learning_rate: float = 0.001
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-07
    amsgrad: bool = False


class Adam2(TensorflowOptimizer):
    """Adam2"""

    parameter: Adam2_Parameters = Adam2_Parameters()

    def set_algorithm(self):
        self.algorithm = tf.keras.optimizers.Adam(
            learning_rate=self.parameter.learning_rate,
            beta_1=self.parameter.beta_1,
            beta_2=self.parameter.beta_2,
            epsilon=self.parameter.epsilon,
            amsgrad=self.parameter.amsgrad,
        )
