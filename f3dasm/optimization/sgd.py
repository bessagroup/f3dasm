from dataclasses import dataclass

from ..base.function import Function
from ..base.optimization import Optimizer, OptimizerParameters


@dataclass
class SGD_Parameters(OptimizerParameters):
    """Hyperparameters for SGD optimizer"""

    learning_rate: float = 1e-2


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
