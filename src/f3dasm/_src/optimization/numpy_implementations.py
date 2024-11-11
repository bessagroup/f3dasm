"""
Optimizers based from the numpy library
"""

#                                                                       Modules
# =============================================================================

# Standard
from typing import Optional, Tuple

# Third-party core
import numpy as np

# Locals
from ..design.domain import Domain
from .optimizer import Optimizer, OptimizerTuple

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class RandomSearch(Optimizer):
    """Naive random search"""
    require_gradients: bool = False

    def __init__(self, domain: Domain, seed: Optional[int] = None, **kwargs):
        self.domain = domain
        self.seed = seed
        self.algorithm = np.random.default_rng(self.seed)

    def update_step(self) -> Tuple[np.ndarray, np.ndarray]:
        x_new = np.atleast_2d(
            [
                self.algorithm.uniform(
                    low=self.domain.get_bounds()[d, 0],
                    high=self.domain.get_bounds()[d, 1])
                for d in range(len(self.domain))
            ]
        )

        # return the data
        return x_new, None


def random_search(seed: Optional[int] = None, **kwargs) -> OptimizerTuple:
    """
    Random search optimizer

    Parameters
    ----------
    seed : int, optional
        Random seed, by default None

    Returns
    -------
    OptimizerTuple
        Optimizer tuple
    """
    return OptimizerTuple(
        base_class=RandomSearch,
        algorithm=None,
        hyperparameters={'seed': seed}
    )
