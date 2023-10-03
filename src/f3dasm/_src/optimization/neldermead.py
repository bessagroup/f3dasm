"""
Nelder-Mead optimizer
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
class NelderMead_Parameters(OptimizerParameters):
    """Hyperparameters for NelderMead optimizer"""

    xatol: float = 0.0
    fatol: float = 0.0
    adaptive: bool = False


class NelderMead(_SciPyOptimizer):
    """Nelder-Mead"""

    method: str = "Nelder-Mead"
    hyperparameters: NelderMead_Parameters = NelderMead_Parameters()

    def get_info(self) -> List[str]:
        return ['Fast', 'Global', 'First-Order', 'Single-Solution']
