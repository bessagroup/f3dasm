"""
Module for Benchmark functions
"""
#                                                                       Modules
# =============================================================================

# Local
from .._src.datageneration.functions import (FUNCTIONS, FUNCTIONS_2D,
                                             FUNCTIONS_7D, find_function,
                                             get_functions, pybenchfunction)
from .._src.datageneration.functions.adapters.augmentor import (
    FunctionAugmentor, Noise, Offset, Scale)
from .._src.datageneration.functions.function import Function
from .._src.datageneration.functions.pybenchfunction import *  # NOQA

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
    'Function',
    'FunctionAugmentor',
    'Noise',
    'Offset',
    'Scale',
    'find_function',
    'get_functions',
    'pybenchfunction',
    *pybenchfunction.__all__,
]
