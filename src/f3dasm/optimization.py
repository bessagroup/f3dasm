"""
Module for optimization
"""
#                                                                       Modules
# =============================================================================

# Local
from ._src.optimization.cg import CG, CG_Parameters
from ._src.optimization.lbfgsb import LBFGSB, LBFGSB_Parameters
from ._src.optimization.neldermead import NelderMead, NelderMead_Parameters
from ._src.optimization.optimizer import Optimizer, OptimizerParameters
from ._src.optimization.optimizer_factory import OPTIMIZERS
from ._src.optimization.randomsearch import (RandomSearch,
                                             RandomSearch_Parameters)

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
