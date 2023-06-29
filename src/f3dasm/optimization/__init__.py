"""
Some API information about the opitmizers
"""
#                                                                       Modules
# =============================================================================

# Standard
import sys
from itertools import chain
from os import path
from typing import TYPE_CHECKING

# Local
from .._imports import _IntegrationModule

if TYPE_CHECKING:
    from .all_optimizers import OPTIMIZERS
    from .cg import CG, CG_Parameters
    from .lbfgsb import LBFGSB, LBFGSB_Parameters
    from .neldermead import NelderMead, NelderMead_Parameters
    from .optimizer import Optimizer
    from .randomsearch import RandomSearch, RandomSearch_Parameters
    from .utils import (create_optimizer_from_dict, create_optimizer_from_json,
                        find_optimizer)


#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

_import_structure: dict = {
    "utils": ["create_optimizer_from_json", "create_optimizer_from_dict", "find_optimizer"],
    "optimizer": ["Optimizer"],
    "cg": ["CG", "CG_Parameters"],
    "lbfgsb": ["LBFGSB", "LBFGSB_Parameters"],
    "neldermead": ["NelderMead", "NelderMead_Parameters"],
    "randomsearch": ["RandomSearch", "RandomSearch_Parameters"],
    "all_optimizers": ["OPTIMIZERS"],
}

if not TYPE_CHECKING:
    class _LocalIntegrationModule(_IntegrationModule):
        __file__ = globals()["__file__"]
        __path__ = [path.dirname(__file__)]
        __all__ = list(chain.from_iterable(_import_structure.values()))
        _import_structure = _import_structure

    sys.modules[__name__] = _LocalIntegrationModule(__name__)
