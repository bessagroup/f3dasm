#                                                                       Modules
# =============================================================================

# Local
from .domain import Domain, make_nd_continuous_domain
from .experimentdata import ExperimentData
from .parameter import (PARAMETERS, CategoricalParameter, ConstantParameter,
                        ContinuousParameter, DiscreteParameter, Parameter)
from .design import Design

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
    'ExperimentData',
    'PARAMETERS',
    'CategoricalParameter',
    'ConstantParameter',
    'ContinuousParameter',
    'DiscreteParameter',
    'Parameter',
    'Design'
]
