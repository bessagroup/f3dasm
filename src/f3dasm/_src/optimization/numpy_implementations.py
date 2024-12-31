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
from ..datageneration.datagenerator import DataGenerator
from .optimizer import ExperimentData, Optimizer

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class NumpyOptimizer(Optimizer):
    """Numpy optimizer class"""
    require_gradients: bool = False

    def __init__(self, seed: Optional[int], **hyperparameters):
        self.seed = seed
        self.hyperparameters = hyperparameters

    def init(self, data: ExperimentData, data_generator: DataGenerator):
        self.data_generator = data_generator
        self.data = data
        self.algorithm = np.random.default_rng(self.seed)

    def update_step(self) -> Tuple[np.ndarray, np.ndarray]:
        x_new = np.atleast_2d(
            [
                self.algorithm.uniform(
                    low=self.data.domain.get_bounds()[d, 0],
                    high=self.data.domain.get_bounds()[d, 1])
                for d in range(len(self.data.domain))
            ]
        )

        # return the data
        return x_new, None


def random_search(seed: Optional[int] = None, **kwargs) -> Optimizer:
    """
    Random search optimizer

    Parameters
    ----------
    seed : int, optional
        Random seed, by default None

    Returns
    -------
    Optimizer
        Optimizer object.
    """
    return NumpyOptimizer(
        seed=seed,
        **kwargs
    )
