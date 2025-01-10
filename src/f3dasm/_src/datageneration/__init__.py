"""
Module for data-generation
"""
#                                                                       Modules
# =============================================================================

# Standard
from typing import List

# Local
from ..core import DataGenerator
from .datagenerator_factory import _datagenerator_factory

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
    '_datagenerator_factory'
]
