"""
Module for optimization
"""
#                                                                       Modules
# =============================================================================

# Standard
from ._imports import try_import

# Local
from .optimizer_factory import create_optimizer

with try_import() as _optuna_imports:
    from .optuna_implementations import tpesampler

with try_import() as _scipy_imports:
    from .scipy_implementations import cg, lbfgsb, nelder_mead

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================
#
# =============================================================================


# =============================================================================

__all__ = [
    "create_optimizer",
    "OptunaOptimizer",
    "tpesampler",
    "cg",
    "lbfgsb",
    "nelder_mead",
]
