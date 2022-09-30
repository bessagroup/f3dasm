from dataclasses import dataclass
import autograd.numpy as np

from ..base.function import Function
from ..base.optimization import Optimizer, OptimizerParameters


@dataclass
class RandomSearch_Parameters(OptimizerParameters):
    """Hyperparameters for RandomSearch optimizer"""

    pass


class RandomSearch(Optimizer):
    """Naive random search"""

    parameter: RandomSearch_Parameters = RandomSearch_Parameters()

    def update_step(self, function: Function) -> None:

        x_new = np.atleast_2d(
            [
                np.random.uniform(low=function.scale_bounds[d, 0], high=function.scale_bounds[d, 1])
                for d in range(function.dimensionality)
            ]
        )

        self.data.add_numpy_arrays(input=x_new, output=function(x_new))
