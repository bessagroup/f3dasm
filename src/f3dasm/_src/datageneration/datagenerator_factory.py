"""
Factory method for creating DataGenerator objects from strings, functions, or
DataGenerator objects.
"""

#                                                                       Modules
# =============================================================================


from __future__ import annotations

# Standard
import inspect
from typing import Callable, Dict, List, Optional

# Local
from .datagenerator import DataGenerator, convert_function
from .functions import _DATAGENERATORS

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================
#
# =============================================================================


DATAGENERATOR_MAPPING: Dict[str, DataGenerator] = {
    f.name.lower().replace(' ', '').replace('-', '').replace(
        '_', '').replace('.', ''): f for f in _DATAGENERATORS}

# =============================================================================


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

        if filtered_name in DATAGENERATOR_MAPPING:
            return DATAGENERATOR_MAPPING[filtered_name](**kwargs)

        else:
            raise KeyError(f"Unknown data generator name: {data_generator}")

    # If the data generator is not a known type, raise an error
    else:
        raise TypeError(f"Unknown data generator type: {type(data_generator)}")

# =============================================================================
