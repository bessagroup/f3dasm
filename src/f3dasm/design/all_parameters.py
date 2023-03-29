#                                                                       Modules
# =============================================================================

# Local
from .parameter import (CategoricalParameter, ConstantParameter,
                        ContinuousParameter, DiscreteParameter)

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


PARAMETERS = [CategoricalParameter, ConstantParameter, ContinuousParameter, DiscreteParameter]
