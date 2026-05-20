"""
Module for optimization
"""
#                                                                       Modules
# =============================================================================

# Local
from ._src.optimization._imports import try_import

with try_import() as _scipy_imports:
    from ._src.optimization import cg, lbfgsb, nelder_mead

with try_import() as _optuna_imports:
    from ._src.optimization import tpesampler

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================
#
# =============================================================================

__all__ = [
    "cg",
    "lbfgsb",
    "nelder_mead",
    "tpesampler",
]
