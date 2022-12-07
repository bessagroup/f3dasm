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
class LBFGSB_Parameters(OptimizerParameters):
    """Hyperparameters for LBFGSB optimizer"""

    ftol: float = 0.0
    gtol: float = 0.0


class LBFGSB(SciPyMinimizeOptimizer):
    """L-BFGS-B"""

    method: str = "L-BFGS-B"
    parameter: LBFGSB_Parameters = LBFGSB_Parameters()
