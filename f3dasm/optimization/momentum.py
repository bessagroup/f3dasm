from dataclasses import dataclass

from ..base.function import Function
from ..base.optimization import Optimizer, OptimizerParameters


@dataclass
class Momentum_Parameters(OptimizerParameters):
    """Hyperparameters for Momentum optimizer"""

    learning_rate: float = 1e-2
    beta: float = 0.9


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
