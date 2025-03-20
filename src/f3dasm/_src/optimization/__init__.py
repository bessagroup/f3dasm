"""
Module for optimization
"""
#                                                                       Modules
# =============================================================================

# Standard
from typing import Callable, List

from .numpy_implementations import random_search
from .optimizer_factory import available_optimizers, create_optimizer
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
    'create_optimizer',
    'cg',
    'lbfgsb',
    'nelder_mead',
    'random_search',
    'available_optimizers',
]
