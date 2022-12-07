#                                                                       Modules
# =============================================================================

# Standard
from dataclasses import dataclass

# Locals
from ..base.optimization import OptimizerParameters
from .adapters.scipy_implementations import SciPyMinimizeOptimizer

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


class NelderMead(SciPyMinimizeOptimizer):
    """Nelder-Mead"""

    method: str = "Nelder-Mead"
    parameter: NelderMead_Parameters = NelderMead_Parameters()
