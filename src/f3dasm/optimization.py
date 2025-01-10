"""
Module for optimization
"""
#                                                                       Modules
# =============================================================================

# Local
from ._src.core import Optimizer
from ._src.optimization import cg, lbfgsb, nelder_mead, random_search
from ._src.optimization.optimizer_factory import available_optimizers

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

__all__ = [
    'Optimizer',
    'available_optimizers',
    'cg',
    'lbfgsb',
    'nelder_mead',
    'random_search'
]
