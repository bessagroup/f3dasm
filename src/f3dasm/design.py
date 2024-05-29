"""
Module for design-of-experiments
"""
#                                                                       Modules
# =============================================================================

# Local
from ._src.design.domain import Domain, make_nd_continuous_domain
from ._src.experimentdata._jobqueue import NoOpenJobsError, Status

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
    'NoOpenJobsError',
    'Status',
]
