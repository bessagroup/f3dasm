#                                                                       Modules
# =============================================================================

# Local
from .constraint import Constraint
from .design import DesignSpace, make_nd_continuous_design
from .experimentdata import ExperimentData
from .parameter import (CategoricalParameter, ConstantParameter,
                        ContinuousParameter, DiscreteParameter, Parameter, PARAMETERS)


#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================
