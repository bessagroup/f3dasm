"""
Optimizers based from the numpy library
"""

#                                                                       Modules
# =============================================================================

# Standard
from typing import Optional

# Third-party core
import numpy as np

# Locals
from ..core import Block, ExperimentData

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class NumpyOptimizer(Block):
    """Numpy optimizer class"""
    require_gradients: bool = False

    def __init__(self, seed: Optional[int], **hyperparameters):
        self.seed = seed
        self.hyperparameters = hyperparameters
        self.algorithm = np.random.default_rng(seed)

    def call(self, data: ExperimentData, **kwargs) -> ExperimentData:
        x_new = np.atleast_2d(
            [
                self.algorithm.uniform(
                    low=data.domain.get_bounds()[d, 0],
                    high=data.domain.get_bounds()[d, 1])
                for d in range(len(data.domain.input_space))
            ]
        )

        # return the data
        return type(data)(domain=data.domain._copy(),
                          input_data=x_new,
                          )


def random_search(seed: Optional[int] = None, **kwargs) -> Block:
    """
    Random search optimizer

    Parameters
    ----------
    seed : int, optional
        Random seed, by default None

    Returns
    -------
    Block
        Optimizer object.
    """
    return NumpyOptimizer(
        seed=seed,
        **kwargs
    )
