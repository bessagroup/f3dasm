
#                                                                       Modules
# =============================================================================

# Standard
from typing import List

# Local
from .._imports import try_import
from . import abaqus
from .datagenerator import DataGenerator
from .functions import pybenchfunction
from .functions.adapters.augmentor import (FunctionAugmentor, Noise, Offset,
                                           Scale)
from .functions.function import Function
from .functions.pybenchfunction import *

# Try importing f3dasm_optimize package
with try_import('f3dasm_simulate') as _imports:
    import f3dasm_simulate

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

# List of available optimizers
DATAGENERATORS: List[DataGenerator] = []

__all__ = [
    'DataGenerator',
    'Function',
    'FunctionAugmentor',
    'Noise',
    'Offset',
    'Scale',
    *pybenchfunction.__all__
]

# Add the optimizers from f3dasm_optimize if applicable
if _imports.is_successful():
    pass
