"""
Optimizers based from the scipy.optimize library
"""

#                                                                       Modules
# =============================================================================

# Standard
from typing import List

# Locals
from ..design.domain import Domain
from .adapters.scipy_implementations import _SciPyOptimizer

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class CG(_SciPyOptimizer):
    """CG"""
    require_gradients: bool = True

    def __init__(self, domain: Domain, gtol: float = 0.0, **kwargs):
        super().__init__(
            domain=domain, method='CG', gtol=gtol)
        self.gtol = gtol

    def _get_info(self) -> List[str]:
        return ['Stable', 'First-Order', 'Single-Solution']

# =============================================================================


class LBFGSB(_SciPyOptimizer):
    """L-BFGS-B"""
    require_gradients: bool = True

    def __init__(self, domain: Domain,
                 ftol: float = 0.0, gtol: float = 0.0, **kwargs):
        super().__init__(
            domain=domain, method='L-BFGS-B', ftol=ftol, gtol=gtol)
        self.ftol = ftol
        self.gtol = gtol

    def _get_info(self) -> List[str]:
        return ['Stable', 'First-Order', 'Single-Solution']

# =============================================================================


class NelderMead(_SciPyOptimizer):
    """Nelder-Mead"""
    require_gradients: bool = False

    def __init__(self, domain: Domain,
                 xatol: float = 0.0, fatol: float = 0.0,
                 adaptive: bool = False, **kwargs):
        super().__init__(
            domain=domain, method='Nelder-Mead', xatol=xatol, fatol=fatol,
            adaptive=adaptive)
        self.xatol = xatol
        self.fatol = fatol
        self.adaptive = adaptive

    def _get_info(self) -> List[str]:
        return ['Fast', 'Global', 'First-Order', 'Single-Solution']
