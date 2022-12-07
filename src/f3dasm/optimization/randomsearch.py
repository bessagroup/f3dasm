#                                                                       Modules
# =============================================================================

# Standard
from dataclasses import dataclass
from typing import Tuple

# Third-party
import autograd.numpy as np

# Locals
from ..base.function import Function
from ..base.optimization import Optimizer, OptimizerParameters

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


@dataclass
class RandomSearch_Parameters(OptimizerParameters):
    """Hyperparameters for RandomSearch optimizer"""

    pass


class RandomSearch(Optimizer):
    """Naive random search"""

    parameter: RandomSearch_Parameters = RandomSearch_Parameters()

    def update_step(self, function: Function) -> Tuple[np.ndarray, np.ndarray]:

        x_new = np.atleast_2d(
            [
                np.random.uniform(
                    low=function.scale_bounds[d, 0], high=function.scale_bounds[d, 1])
                for d in range(function.dimensionality)
            ]
        )

        return x_new, function(x_new)
        # self.data.add_numpy_arrays(input=x_new, output=function(x_new))
