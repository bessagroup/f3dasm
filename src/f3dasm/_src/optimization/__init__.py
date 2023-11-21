"""
Module for optimization
"""
#                                                                       Modules
# =============================================================================

# Standard
from typing import List

# Local
from .cg import CG, CG_Parameters
from .lbfgsb import LBFGSB, LBFGSB_Parameters
from .neldermead import NelderMead, NelderMead_Parameters
from .optimizer import Optimizer
from .randomsearch import RandomSearch, RandomSearch_Parameters

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

# List of available optimizers
_OPTIMIZERS: List[Optimizer] = [RandomSearch, CG, LBFGSB, NelderMead]


__all__ = [
    'CG',
    'CG_Parameters',
    'LBFGSB',
    'LBFGSB_Parameters',
    'NelderMead',
    'NelderMead_Parameters',
    'Optimizer',
    'RandomSearch',
    'RandomSearch_Parameters',
    '_OPTIMIZERS',
    'find_optimizer',
]


def find_optimizer(query: str) -> Optimizer:
    """Find a optimizer from the f3dasm.optimizer submodule

    Parameters
    ----------
    query
        string representation of the requested optimizer

    Returns
    -------
        class of the requested optimizer
    """
    try:
        return list(filter(
            lambda optimizer: optimizer.__name__ == query, _OPTIMIZERS))[0]
    except IndexError:
        return ValueError(f'Optimizer {query} not found!')
