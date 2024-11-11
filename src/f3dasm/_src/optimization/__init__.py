"""
Module for optimization
"""
#                                                                       Modules
# =============================================================================

# Standard
from typing import Callable, List

# Local
from .numpy_implementations import random_search
from .optimizer import Optimizer
from .scipy_implementations import cg, lbfgsb, nelder_mead

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

# List of available optimizers
_OPTIMIZERS: List[Callable] = [
    cg, lbfgsb, nelder_mead, random_search]


__all__ = [
    'find_optimizer',
    'random_search',
    'cg',
    'lbfgsb',
    'nelder_mead',
    'Optimizer',
]


def find_optimizer(query: str) -> Callable:
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
