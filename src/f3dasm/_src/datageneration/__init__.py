"""
Module for data-generation
"""
#                                                                       Modules
# =============================================================================

# Standard

# Local
from ..datagenerator import DataGenerator
from .datagenerator_factory import create_datagenerator

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

__all__ = [
    'DataGenerator',
    'create_datagenerator'
]
