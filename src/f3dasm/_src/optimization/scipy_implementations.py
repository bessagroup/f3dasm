"""
Optimizers based from the scipy.optimize library
"""

#                                                                       Modules
# =============================================================================

# Standard
from dataclasses import dataclass
from typing import List

# Locals
from .adapters.scipy_implementations import _SciPyOptimizer
from .optimizer import OptimizerParameters

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


@dataclass
class CG_Parameters(OptimizerParameters):
    """CG Parameters"""

    gtol: float = 0.0


class CG(_SciPyOptimizer):
    """CG"""
    require_gradients: bool = True
    method: str = "CG"
    hyperparameters: CG_Parameters = CG_Parameters()

    def get_info(self) -> List[str]:
        return ['Stable', 'First-Order', 'Single-Solution']

# =============================================================================


@dataclass
class LBFGSB_Parameters(OptimizerParameters):
    """Hyperparameters for LBFGSB optimizer"""

    ftol: float = 0.0
    gtol: float = 0.0


class LBFGSB(_SciPyOptimizer):
    """L-BFGS-B"""
    require_gradients: bool = True
    method: str = "L-BFGS-B"
    hyperparameters: LBFGSB_Parameters = LBFGSB_Parameters()

    def get_info(self) -> List[str]:
        return ['Stable', 'First-Order', 'Single-Solution']

# =============================================================================


@dataclass
class NelderMead_Parameters(OptimizerParameters):
    """Hyperparameters for NelderMead optimizer"""

    xatol: float = 0.0
    fatol: float = 0.0
    adaptive: bool = False


class NelderMead(_SciPyOptimizer):
    """Nelder-Mead"""
    require_gradients: bool = False
    method: str = "Nelder-Mead"
    hyperparameters: NelderMead_Parameters = NelderMead_Parameters()

    def get_info(self) -> List[str]:
        return ['Fast', 'Global', 'First-Order', 'Single-Solution']
