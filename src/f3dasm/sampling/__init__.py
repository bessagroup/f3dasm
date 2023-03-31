#                                                                       Modules
# =============================================================================

# Standard
import sys
from itertools import chain
from os import path
from typing import TYPE_CHECKING

# Locals
from .._imports import _IntegrationModule

if TYPE_CHECKING:
    from .all_samplers import SAMPLERS
    from .latinhypercube import LatinHypercube
    from .randomuniform import RandomUniform
    from .sampler import Sampler
    from .sobolsequence import SobolSequence
    from .utils import create_sampler_from_json, find_sampler

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

_import_structure: dict = {
    "utils": ["create_sampler_from_json", "find_sampler"],
    "sampler": ["Sampler"],
    "latinhypercube": ["LatinHypercube"],
    "randomuniform": ["RandomUniform"],
    "sobolsequence": ["SobolSequence"],
    "all_samplers": ["SAMPLERS"],
}

if not TYPE_CHECKING:
    class _LocalIntegrationModule(_IntegrationModule):
        __file__ = globals()["__file__"]
        __path__ = [path.dirname(__file__)]
        __all__ = list(chain.from_iterable(_import_structure.values()))
        _import_structure = _import_structure

    sys.modules[__name__] = _LocalIntegrationModule(__name__)
