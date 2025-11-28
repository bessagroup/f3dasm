"""
Module for optimization
"""
#                                                                       Modules
# =============================================================================

# Local
from ._src.optimization import cg, lbfgsb, nelder_mead, tpesampler

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
