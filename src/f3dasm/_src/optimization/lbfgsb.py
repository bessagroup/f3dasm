"""
L-BFGS-B optimizer
"""

#                                                                       Modules
# =============================================================================

# Standard
from dataclasses import dataclass
from typing import List

from .adapters.scipy_implementations import _SciPyOptimizer
# Locals
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
class LBFGSB_Parameters(OptimizerParameters):
    """Hyperparameters for LBFGSB optimizer"""

    ftol: float = 0.0
    gtol: float = 0.0


class LBFGSB(_SciPyOptimizer):
    """L-BFGS-B"""

    method: str = "L-BFGS-B"
    hyperparameters: LBFGSB_Parameters = LBFGSB_Parameters()

    def get_info(self) -> List[str]:
        return ['Stable', 'First-Order', 'Single-Solution']
