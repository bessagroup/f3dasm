"""
Random Search optimizer
"""

#                                                                       Modules
# =============================================================================

# Standard
from dataclasses import dataclass
from typing import List, Tuple

# Third-party core
import autograd.numpy as np

# Locals
from ..datageneration.datagenerator import DataGenerator
from .optimizer import Optimizer, OptimizerParameters

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

    hyperparameters: RandomSearch_Parameters = RandomSearch_Parameters()

    def set_seed(self):
        np.random.seed(self.seed)

    def update_step(self, data_generator: DataGenerator) -> Tuple[np.ndarray, np.ndarray]:
        # BUG: This setting of seed results in the same value being samples all the time!
        # self.set_seed()

        x_new = np.atleast_2d(
            [
                np.random.uniform(
                    low=self.domain.get_bounds()[d, 0], high=self.domain.get_bounds()[d, 1])
                for d in range(len(self.domain))
            ]
        )

        # return the data
        return x_new, None

    def get_info(self) -> List[str]:
        return ['Fast', 'Single-Solution']
