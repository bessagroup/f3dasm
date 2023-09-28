"""
Module for data-generation
"""
#                                                                       Modules
# =============================================================================

# Local
from .._src.datageneration.abaqus.abaqus_simulator import AbaqusSimulator
from .._src.datageneration.datagenerator import DataGenerator
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
    'AbaqusSimulator',
    'DataGenerator',
    'Function',
    'FunctionAugmentor',
    'FUNCTIONS',
    'FUNCTIONS_2D',
    'FUNCTIONS_7D',
    'Noise',
    'Offset',
    'pybenchfunction',
    'Scale',
    'find_function',
    'get_functions',
    *pybenchfunction.__all__
]