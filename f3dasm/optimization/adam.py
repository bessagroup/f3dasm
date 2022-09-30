from dataclasses import dataclass

import autograd.numpy as np

from ..base.function import Function
from ..base.optimization import Optimizer, OptimizerParameters


@dataclass
class Adam_Parameters(OptimizerParameters):
    """Hyperparameters for Adam optimizer

    Args:
        learning_rate (float): (Default = 1e-2)
        beta_1 (float): (Default = 0.9)
        beta_2 (float): (Default = 0.999)
        epsilon (float): (Default = 1e-8)
    """

    learning_rate: float = 1e-2
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-8


class Adam(Optimizer):
    """Gradient-based Adam optimizer"""

    parameter: Adam_Parameters = Adam_Parameters()

    def init_parameters(self):
        self.m = 0
        self.v = 0

    def update_step(self, function: Function) -> None:

        x = self.data.get_input_data().iloc[-self.parameter.population :].to_numpy()

        g = function.dfdx(x)
        t = self.data.get_number_of_datapoints()
        m = self.parameter.beta_1 * self.m + (1 - self.parameter.beta_2) * g
        v = self.parameter.beta_2 * self.v + (1 - self.parameter.beta_2) * np.power(g, 2)

        m_hat = m / (1 - np.power(self.parameter.beta_1, t))
        v_hat = v / (1 - np.power(self.parameter.beta_2, t))
        x_new = x - self.parameter.learning_rate * m_hat / (np.sqrt(v_hat) + self.parameter.epsilon)

        # Force bounds
        if self.parameter.force_bounds:
            x_new = self._force_bounds(x_new, function.scale_bounds)

        # Update dynamic parameters
        self.m = m
        self.v = v

        self.data.add_numpy_arrays(input=x_new, output=function(x_new))
