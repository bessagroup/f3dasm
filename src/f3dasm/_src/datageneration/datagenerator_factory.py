"""
Factory method for creating DataGenerator objects from strings, functions, or
DataGenerator objects.
"""

#                                                                       Modules
# =============================================================================


from __future__ import annotations

# Standard
import inspect
from functools import partial
from typing import Any, Callable, Dict, List, Optional

# Local
from ..core import DataGenerator
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


def convert_function(f: Callable,
                     output: Optional[List[str]] = None,
                     kwargs: Optional[Dict[str, Any]] = None,
                     to_disk: Optional[List[str]] = None) -> DataGenerator:
    """
    Converts a given function `f` into a `DataGenerator` object.

    Parameters
    ----------
    f : Callable
        The function to be converted.
    output : Optional[List[str]], optional
        A list of names for the return values of the function.
        Defaults to None.
    kwargs : Optional[Dict[str, Any]], optional
        Additional keyword arguments passed to the function. Defaults to None.
    to_disk : Optional[List[str]], optional
        The list of output names where the value needs to be stored on disk.
        Defaults to None.

    Returns
    -------
    DataGenerator
        A converted `DataGenerator` object.

    Notes
    -----

    The function `f` can have any number of arguments and any number of returns
    as long as they are consistent with the `input` and `output` arguments that
    are given to this function.
    """
    signature = inspect.signature(f)
    input = list(signature.parameters)
    kwargs = kwargs if kwargs is not None else {}
    to_disk = to_disk if to_disk is not None else []
    output = output if output is not None else []

    class TempDataGenerator(DataGenerator):
        def execute(self, **_kwargs) -> None:
            _input = {input_name:
                      self.experiment_sample.input_data.get(input_name)
                      for input_name in input if input_name not in kwargs}
            _output = f(**_input, **kwargs)

            # check if output is empty
            if output is None:
                return

            if len(output) == 1:
                _output = (_output,)

            for name, value in zip(output, _output):
                if name in to_disk:
                    self.experiment_sample.store(name=name,
                                                 object=value,
                                                 to_disk=True)
                else:
                    self.experiment_sample.store(name=name,
                                                 object=value,
                                                 to_disk=False)

    return TempDataGenerator()


def _datagenerator_factory(
        data_generator: str | Callable | DataGenerator,
        output_names: Optional[List[str]] = None, **kwargs) -> DataGenerator:

    # If the data generator is already a DataGenerator object, return it
    if isinstance(data_generator, DataGenerator):
        return data_generator

    # If the data generator is a function, convert it to a DataGenerator object
    if inspect.isfunction(data_generator) or isinstance(
            data_generator, partial):
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
