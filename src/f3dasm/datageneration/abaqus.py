"""
Module for data-generation with ABAQUS Simulation software
"""
#                                                                       Modules
# =============================================================================

# Local
from .._src.datageneration.abaqus.abaqus_functions import (post_process,
                                                           pre_process)
from .._src.datageneration.abaqus.abaqus_simulator import AbaqusSimulator

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

__all__ = [
    'AbaqusSimulator',
    'post_process',
    'pre_process',
]
