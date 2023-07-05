#                                                                       Modules
# =============================================================================

# Standard
from typing import List

# Locals
from .latinhypercube import LatinHypercube
from .randomuniform import RandomUniform
from .sampler import Sampler
from .sobolsequence import SobolSequence

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

# List of available samplers
SAMPLERS: List[Sampler] = [RandomUniform, LatinHypercube, SobolSequence]


def find_sampler(query: str) -> Sampler:
    """Find a Sampler from the f3dasm.design submodule

    Parameters
    ----------
    query
        string representation of the requested sampler

    Returns
    -------
        class of the requested sampler
    """
    try:
        return list(filter(lambda parameter: parameter.__name__ == query, SAMPLERS))[0]
    except IndexError:
        return ValueError(f'Sampler {query} not found!')


__all__ = [
    'LatinHypercube',
    'RandomUniform',
    'Sampler',
    'SobolSequence',
    'SAMPLERS',
    'find_sampler'
]
