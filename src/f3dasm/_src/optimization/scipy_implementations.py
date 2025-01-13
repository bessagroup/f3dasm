"""
Optimizers based from the scipy.optimize library
"""

#                                                                       Modules
# =============================================================================

# Locals
from ..core import Block
from .adapters.scipy_implementations import ScipyOptimizer

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


def cg(gtol: float = 0.0, **kwargs) -> Block:
    """
    Conjugate Gradient optimizer
    Adapted from scipy.optimize.minimize

    Parameters
    ----------
    gtol : float, optional
        Gradient norm tolerance, by default 0.0

    Returns
    -------
    Optimizer
        Optimizer
    """
    return ScipyOptimizer(
        algorithm='CG',
        gtol=gtol,
        **kwargs
    )

# =============================================================================


def lbfgsb(ftol: float = 0.0, gtol: float = 0.0, **kwargs) -> Block:
    """
    L-BFGS-B optimizer
    Adapted from scipy.optimize.minimize

    Parameters
    ----------
    ftol : float, optional
        Function value tolerance, by default 0.0
    gtol : float, optional
        Gradient norm tolerance, by default 0.0

    Returns
    -------
    Optimizer
        Optimizer
    """
    return ScipyOptimizer(
        algorithm='L-BFGS-B',
        ftol=ftol,
        gtol=gtol,
        **kwargs
    )

# =============================================================================


def nelder_mead(xatol: float = 0.0, fatol: float = 0.0,
                adaptive: bool = False, **kwargs) -> Block:
    """
    Nelder-Mead optimizer
    Adapted from scipy.optimize.minimize

    Parameters
    ----------
    xatol : float, optional
        Absolute error in xopt between iterations, by default 0.0
    fatol : float, optional
        Absolute error in fun(xopt) between iterations, by default 0.0
    adaptive : bool, optional
        Adapt the algorithm, by default False

    Returns
    -------
    Optimizer
        Optimizer
    """
    return ScipyOptimizer(
        algorithm='Nelder-Mead',
        xatol=xatol,
        fatol=fatol,
        adaptive=adaptive,
        **kwargs
    )
