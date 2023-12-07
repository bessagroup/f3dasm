"""
Module for design-of-experiments
"""
#                                                                       Modules
# =============================================================================

# Local
from ._src.design.domain import Domain, make_nd_continuous_domain
from ._src.design.parameter import (PARAMETERS, _CategoricalParameter,
                                    _ConstantParameter, _ContinuousParameter,
                                    _DiscreteParameter, _Parameter)
from ._src.experimentdata._data import _Data
from ._src.experimentdata._jobqueue import NoOpenJobsError, Status, _JobQueue

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

__all__ = [
    '_CategoricalParameter',
    '_ConstantParameter',
    '_ContinuousParameter',
    '_DiscreteParameter',
    'Domain',
    'make_nd_continuous_domain',
    'NoOpenJobsError',
    'PARAMETERS',
    '_Parameter',
    'Status',
    '_Data',
    '_JobQueue',
]
