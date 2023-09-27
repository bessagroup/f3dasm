"""
Module for sampling
"""

#                                                                       Modules
# =============================================================================

# Locals
from ._src.sampling import SAMPLERS, find_sampler
from ._src.sampling.latinhypercube import LatinHypercube
from ._src.sampling.randomuniform import RandomUniform
from ._src.sampling.sampler import Sampler
from ._src.sampling.sobolsequence import SobolSequence

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

__all__ = [
    'Sampler',
    'SAMPLERS',
    'find_sampler',
    'LatinHypercube',
    'RandomUniform',
    'SobolSequence',
]
