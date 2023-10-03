"""
Module for optimization
"""
#                                                                       Modules
# =============================================================================

# Local
# from ._src._imports import try_import
from ._src.optimization.cg import CG, CG_Parameters
from ._src.optimization.lbfgsb import LBFGSB, LBFGSB_Parameters
from ._src.optimization.neldermead import NelderMead, NelderMead_Parameters
from ._src.optimization.optimizer import Optimizer, OptimizerParameters
from ._src.optimization.optimizer_factory import OPTIMIZERS
from ._src.optimization.randomsearch import (RandomSearch,
                                             RandomSearch_Parameters)

# # Try importing f3dasm_optimize package
# with try_import('f3dasm_optimize') as _imports:
#     import f3dasm_optimize


#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

__all__ = [
    'CG',
    'CG_Parameters',
    'LBFGSB',
    'LBFGSB_Parameters',
    'NelderMead',
    'NelderMead_Parameters',
    'Optimizer',
    'OptimizerParameters',
    'OPTIMIZERS',
    'RandomSearch',
    'RandomSearch_Parameters',
]

# # Add the optimizers from f3dasm_optimize if applicable
# if _imports.is_successful():
#     OPTIMIZERS.extend(f3dasm_optimize.OPTIMIZERS)
