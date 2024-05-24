"""
Module for optimization
"""
#                                                                       Modules
# =============================================================================

# Local
from ._src.optimization.optimizer import Optimizer
from ._src.optimization.optimizer_factory import OPTIMIZERS

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
    'OPTIMIZERS',
]
