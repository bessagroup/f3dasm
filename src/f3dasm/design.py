"""
Module for design-of-experiments
"""
#                                                                       Modules
# =============================================================================

# Local
from ._src.core import Sampler
from ._src.design.domain import Domain, make_nd_continuous_domain

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

__all__ = [
    'Domain',
    'make_nd_continuous_domain',
    'Sampler',
]
