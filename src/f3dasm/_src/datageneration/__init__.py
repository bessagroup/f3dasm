"""
Module for data-generation
"""
#                                                                       Modules
# =============================================================================

# Standard
from typing import List

# Local
from .datagenerator import DataGenerator
from .functions import pybenchfunction
from .functions.adapters.augmentor import (FunctionAugmentor, Noise, Offset,
                                           Scale)
from .functions.function import Function
from .functions.pybenchfunction import *  # NOQA

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

try:
    import f3dasm_simulate  # NOQA
except ImportError:
    pass

# List of available optimizers
DATAGENERATORS: List[DataGenerator] = []

__all__ = [
    'DataGenerator',
    'Function',
    'FunctionAugmentor',
    'Noise',
    'Offset',
    'Scale',
    'DATAGENERATORS',
    *pybenchfunction.__all__
]
