"""
Module for Benchmark functions
"""
#                                                                       Modules
# =============================================================================

# Local
from .._src.datageneration.functions import (FUNCTIONS, FUNCTIONS_2D,
                                             FUNCTIONS_7D, get_functions)

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

__all__ = [
    'FUNCTIONS',
    'FUNCTIONS_2D',
    'FUNCTIONS_7D',
    'get_functions',
]
