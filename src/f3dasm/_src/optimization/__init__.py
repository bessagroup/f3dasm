"""
Module for optimization
"""
#                                                                       Modules
# =============================================================================

# Standard
from typing import Callable, List

from ..core import Optimizer
from .numpy_implementations import random_search
from .optimizer_factory import _optimizer_factory, available_optimizers
from .scipy_implementations import cg, lbfgsb, nelder_mead

# Local


#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


# =============================================================================


__all__ = [
    'Optimizer',
    '_optimizer_factory',
    'cg',
    'lbfgsb',
    'nelder_mead',
    'random_search',
    'available_optimizers',
]
