"""
Module for the data generator factory.
"""
#                                                                       Modules
# =============================================================================

from __future__ import annotations

import inspect
from typing import Callable, Dict, List, Optional

from ...design.domain import Domain
from ..datagenerator import DataGenerator, convert_function
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


def _datagenerator_factory(
        data_generator: str | Callable | DataGenerator,
        output_names: Optional[List[str]] = None, **kwargs) -> DataGenerator:

    # If the data generator is already a DataGenerator object, return it
    if isinstance(data_generator, DataGenerator):
        return data_generator

    # If the data generator is a function, convert it to a DataGenerator object
    if inspect.isfunction(data_generator):
        if output_names is None:
            raise TypeError(
                ("If you provide a function as data generator, you have to"
                    "provide the names of the return arguments with the"
                    "output_names attribute."))
        return convert_function(
            f=data_generator, output=output_names, kwargs=kwargs)

    # If the data generator is a string, check if it is a known data generator
    if isinstance(data_generator, str):

        filtered_name = data_generator.lower().replace(
            ' ', '').replace('-', '').replace('_', '').replace('.', '')

        if filtered_name in FUNCTION_MAPPING:
            return FUNCTION_MAPPING[filtered_name](**kwargs)

        else:
            raise KeyError(f"Unknown data generator name: {data_generator}")

    # If the data generator is not a known type, raise an error
    else:
        raise TypeError(f"Unknown data generator type: {type(data_generator)}")


def is_dim_compatible(data_generator: str, domain: Domain) -> bool:
    func = _datagenerator_factory(data_generator)
    return func.is_dim_compatible(len(domain))
