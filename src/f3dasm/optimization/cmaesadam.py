import autograd.numpy as np

from ..base.data import Data
from ..base.metaoptimizer import EqualParts_Strategy, MetaOptimizer
from .adam import Adam
from .cmaes import CMAES


class CMAESAdam(MetaOptimizer):
    """CMAES-Adam Metaoptimizer"""

    def __init__(self, data: Data, seed: int = np.random.randint(low=0, high=1e5), hyperparameters=None):
        optimizers = [
            CMAES(data=data, seed=seed),
            Adam(data=data, seed=seed),
        ]

        strategy = EqualParts_Strategy(optimizers=optimizers)
        super().__init__(data=data, strategy=strategy, seed=seed)
