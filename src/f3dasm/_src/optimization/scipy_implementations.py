"""
Optimizers based from the scipy.optimize library
"""

#                                                                       Modules
# =============================================================================

# Standard
from typing import List

# Locals
from .adapters.scipy_implementations import _SciPyOptimizer

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

CG_DEFAULTS = {'gtol': 0.0}


class CG(_SciPyOptimizer):
    """CG"""
    require_gradients: bool = True
    method: str = "CG"
    default_hyperparameters = CG_DEFAULTS

    def get_info(self) -> List[str]:
        return ['Stable', 'First-Order', 'Single-Solution']

# =============================================================================


LBFGSB_DEFAULTS = {'ftol': 0.0, 'gtol': 0.0}


class LBFGSB(_SciPyOptimizer):
    """L-BFGS-B"""
    require_gradients: bool = True
    method: str = "L-BFGS-B"
    default_hyperparameters = LBFGSB_DEFAULTS

    def get_info(self) -> List[str]:
        return ['Stable', 'First-Order', 'Single-Solution']

# =============================================================================


NelderMead_DEFAULTS = {'xatol': 0.0, 'fatol': 0.0,
                       'adaptive': False}


class NelderMead(_SciPyOptimizer):
    """Nelder-Mead"""
    require_gradients: bool = False
    method: str = "Nelder-Mead"
    default_hyperparameters = NelderMead_DEFAULTS

    def get_info(self) -> List[str]:
        return ['Fast', 'Global', 'First-Order', 'Single-Solution']
