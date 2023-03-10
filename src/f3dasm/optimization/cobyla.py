#                                                                       Modules
# =============================================================================

# Standard
from dataclasses import dataclass
from typing import List

# Locals
from .adapters.scipy_implementations import SciPyMinimizeOptimizer
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
class COBYLA_Parameters(OptimizerParameters):
    """Hyperparameters for COBYLA optimizer"""

    tol: float = 0.0
    rhobeg: float = 1.0
    catol: float = 0.0
    maxiter: int = 1e7


class COBYLA(SciPyMinimizeOptimizer):
    """Minimize a scalar function of one or more variables using the
    Constrained Optimization BY Linear Approximation (COBYLA) algorithm."""

    method: str = "COBYLA"
    parameter: COBYLA_Parameters = COBYLA_Parameters()

    def get_info(self) -> List[str]:
        return ['Fast', 'Derivative-Free', 'Single-Solution']
