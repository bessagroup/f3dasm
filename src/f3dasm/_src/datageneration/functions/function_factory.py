"""
Module for the data generator factory.
"""
#                                                                       Modules
# =============================================================================

from typing import Dict

from ...design.domain import Domain
from ..datagenerator import DataGenerator
from . import FUNCTIONS

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

FUNTION_MAPPING: Dict[str, DataGenerator] = {
    f.name.lower().replace(' ', '').replace('-', '').replace('_', ''): f for f in FUNCTIONS}


def datagenerator_factory(data_generator: str, domain: Domain, kwargs):

    filtered_name = data_generator.lower().replace(' ', '').replace('-', '').replace('_', '')

    if filtered_name in FUNTION_MAPPING:
        return FUNTION_MAPPING[filtered_name](dimensionality=len(domain), **kwargs)

    else:
        raise KeyError(f"Unknown data generator: {data_generator}")
