#                                                                       Modules
# =============================================================================

# Standard
from enum import Enum

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class _Jobs(Enum):
    OPEN = 0
    IN_PROGRESS = 1
    FINISHED = 2
    ERROR = 3
