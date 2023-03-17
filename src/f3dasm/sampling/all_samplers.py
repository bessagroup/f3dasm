#                                                                       Modules
# =============================================================================

# Standard
from typing import List

# Locals
from . import latinhypercube, randomuniform, sobolsequence
from .sampler import Sampler

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

# List of available samplers
SAMPLERS: List[Sampler] = []

# Core samplers
SAMPLERS.append(randomuniform.RandomUniform)

# Extension samplers
if latinhypercube._imports.is_successful():
    SAMPLERS.append(latinhypercube.LatinHypercube)

if sobolsequence._imports.is_successful():
    SAMPLERS.append(sobolsequence.SobolSequence)
