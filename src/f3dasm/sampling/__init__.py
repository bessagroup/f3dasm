#                                                                       Modules
# =============================================================================

# Locals
from .latinhypercube import LatinHypercube
from .randomuniform import RandomUniform
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
SAMPLERS = [LatinHypercube, RandomUniform, SobolSequence]
