"""
Design-of-experiments (DOE) module for the f3dasm package.
"""

#                                                                       Modules
# =============================================================================

# Local
from .domain import Domain, _domain_factory
from .samplers import create_sampler

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

__all__ = ['Domain',
           '_domain_factory',
           'create_sampler'
           ]
