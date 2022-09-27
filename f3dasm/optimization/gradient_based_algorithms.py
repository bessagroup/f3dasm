import numpy as np

from ..optimization.hyperparameters import (
    Adam_Parameters,
    Momentum_Parameters,
    SGD_Parameters,
)

from ..base.optimization import Optimizer
from ..base.function import Function


class SGD(Optimizer):
    """Gradient-based Stochastig Gradient Descent (SGD) optimizer"""

    parameter: SGD_Parameters = SGD_Parameters()

    def update_step(self, function: Function) -> None:

        x = self.data.get_input_data().iloc[-self.parameter.population :].to_numpy()

        g = function.dfdx(x)

        x_new = x - self.parameter.learning_rate * g

        # Force bounds
        if self.parameter.force_bounds:
            x_new = self._force_bounds(x_new, function.scale_bounds)

        self.data.add_numpy_arrays(input=x_new, output=function(x_new))


class Momentum(Optimizer):
    """Gradient-based Momentum optimizer"""

    parameter: Momentum_Parameters = Momentum_Parameters()

    def init_parameters(self):
        self.m = 0

    def update_step(self, function: Function) -> None:

        x = self.data.get_input_data().iloc[-self.parameter.population :].to_numpy()

        g = function.dfdx(x)

        m = self.parameter.beta * self.m + (1 - self.parameter.beta) * g
        x_new = x - self.parameter.learning_rate * m

        # Force bounds
        if self.parameter.force_bounds:
            x_new = self._force_bounds(x_new, function.scale_bounds)

        # Update dynamic parameters
        self.m = m

        self.data.add_numpy_arrays(input=x_new, output=function(x_new))


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
