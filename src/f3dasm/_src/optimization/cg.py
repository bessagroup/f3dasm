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

    method: str = "CG"
    hyperparameters: CG_Parameters = CG_Parameters()

    def get_info(self) -> List[str]:
        return ['Stable', 'First-Order', 'Single-Solution']
