"""
Factory method for creating DataGenerator objects from strings, functions, or
DataGenerator objects.
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
from collections.abc import Iterable

# Local
from ..core import DataGenerator, datagenerator
from .benchmarkfunctions import BENCHMARK_FUNCTIONS

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================
#
# =============================================================================


def create_datagenerator(
    data_generator: str, output_names: str | Iterable[str], **parameters
) -> DataGenerator:
    """
    Create a DataGenerator block from one of the built-in data generators.

    Parameters
    ----------
    data_generator : str | DataGenerator
        name of the built-in data generator. This can be a string with the name
        of the data generator, a Block object (this will just by-pass the
        function), or a function.
    **parameters
        Additional keyword arguments passed when initializing the data
        generator

    Returns
    -------
    DataGenerator
        DataGenerator object

    Raises
    ------
    KeyError
        If the built-in sampler data generator is not recognized.
    TypeError
        If the given type is not recognized.
    """
    # If the data generator is a string, check if it is a known data generator
    if isinstance(data_generator, str):
        filtered_name = (
            data_generator.lower()
            .replace(" ", "")
            .replace("-", "")
            .replace("_", "")
            .replace(".", "")
        )

        if filtered_name in BENCHMARK_FUNCTIONS:
            return datagenerator(output_names=output_names)(
                BENCHMARK_FUNCTIONS[filtered_name]
            )

        else:
            raise KeyError(f"Unknown data generator name: {data_generator}")

    # If the data generator is not a known type, raise an error
    else:
        raise TypeError(f"Unknown data generator type: {type(data_generator)}")


# =============================================================================
