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
class CG_Parameters(OptimizerParameters):
    """CG Parameters"""

    gtol: float = 0.0


class CG(SciPyMinimizeOptimizer):
    """CG"""

    method: str = "CG"
    parameter: CG_Parameters = CG_Parameters()
