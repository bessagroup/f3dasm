"""
Module for the data generator factory.
"""
#                                                                       Modules
# =============================================================================

from __future__ import annotations

from typing import Dict

from ...design.domain import Domain
from ..datagenerator import DataGenerator
from . import _FUNCTIONS

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

FUNCTION_MAPPING: Dict[str, DataGenerator] = {
    f.name.lower().replace(' ', '').replace('-', '').replace(
        '_', '').replace('.', ''): f for f in _FUNCTIONS}


def _datagenerator_factory(data_generator: str, **kwargs) -> DataGenerator:

    filtered_name = data_generator.lower().replace(
        ' ', '').replace('-', '').replace('_', '').replace('.', '')

    if filtered_name in FUNCTION_MAPPING:
        return FUNCTION_MAPPING[filtered_name](**kwargs)

    else:
        raise KeyError(f"Unknown data generator: {data_generator}")


def is_dim_compatible(data_generator: str, domain: Domain) -> bool:
    func = _datagenerator_factory(data_generator)
    return func.is_dim_compatible(len(domain))
