"""
Optimizers based from the scipy.optimize library
"""

#                                                                       Modules
# =============================================================================

# Locals
from .adapters.scipy_implementations import _SciPyOptimizer
from .optimizer import OptimizerTuple

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


# class CG(_SciPyOptimizer):
#     """CG"""
#     require_gradients: bool = True

#     def __init__(self, domain: Domain, gtol: float = 0.0, **kwargs):
#         super().__init__(
#             domain=domain, method='CG', gtol=gtol)
#         self.gtol = gtol

#     def _get_info(self) -> List[str]:
#         return ['Stable', 'First-Order', 'Single-Solution']


def cg(gtol: float = 0.0, **kwargs) -> OptimizerTuple:
    return OptimizerTuple(
        base_class=_SciPyOptimizer,
        algorithm='CG',
        hyperparameters={'gtol': gtol, **kwargs}
    )

# =============================================================================


def lbfgsb(ftol: float = 0.0, gtol: float = 0.0, **kwargs) -> OptimizerTuple:
    return OptimizerTuple(
        base_class=_SciPyOptimizer,
        algorithm='L-BFGS-B',
        hyperparameters={'ftol': ftol, 'gtol': gtol, **kwargs}
    )

# =============================================================================


def nelder_mead(xatol: float = 0.0, fatol: float = 0.0,
                adaptive: bool = False, **kwargs) -> OptimizerTuple:
    return OptimizerTuple(
        base_class=_SciPyOptimizer,
        algorithm='Nelder-Mead',
        hyperparameters={
            'xatol': xatol, 'fatol': fatol, 'adaptive': adaptive, **kwargs}
    )
