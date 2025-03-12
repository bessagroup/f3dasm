"""
Factory method for creating DataGenerator objects from strings, functions, or
DataGenerator objects.
"""

#                                                                       Modules
# =============================================================================


from __future__ import annotations

# Standard
import inspect
from typing import Any, Callable, Dict, Iterable

# Local
from ..core import DataGenerator
from ..experimentsample import ExperimentSample
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


def datagenerator(output_names: Iterable[str]
                  ) -> Callable[[Callable[..., Any]], DataGenerator]:
    """
    Decorator to convert a function into a `DataGenerator` subclass.

    The decorated function should take named arguments matching the keys in
    the Domain and return one or multiple output values. These values will be
    stored in the `ExperimentData` under the names specified in `output_names`.

    Parameters
    ----------
    output_names : Iterable[str]
        A list of names for the returned values of the decorated function.
        The function's return values will be stored in the `ExperimentSample`
        object under these names.

    Returns
    -------
    Callable[[Callable[..., Any]], DataGenerator]
        A decorator that transforms a function into a `DataGenerator` subclass.

    Raises
    ------
    ValueError
        If `output_names` is not provided.

    Examples
    --------
    >>> @datagenerator(output_names=['y'])
    ... def my_function(x0: float, x1: float, x2: float) -> float:
    ...     return x0**2 + x1**2 + x2**2
    ...
    >>> experiment_data = ExperimentData(domaind=domain)
    >>> experiment_data = my_function.call(experiment_data)
    """

    if not output_names:
        raise ValueError((
            "If you provide a function as a data generator, you must "
            "provide the names of the return arguments with the `output_names`"
            "attribute."
        ))

    # If the output names is a single string, convert it to a list
    if isinstance(output_names, str):
        output_names = [output_names]

    def decorator(f: Callable[..., Any]) -> DataGenerator:
        signature = inspect.signature(f)
        input_names = list(signature.parameters.keys())

        class FunctionDataGenerator(DataGenerator):
            """
            Auto-generated DataGenerator subclass from a decorated function.
            """

            def execute(self, experiment_sample: ExperimentSample,
                        **kwargs: Any) -> ExperimentSample:
                """
                Executes the data generation process by calling the decorated
                function.

                Parameters
                ----------
                experiment_sample : ExperimentSample
                    The experiment sample containing input data.
                **kwargs : dict
                    Additional keyword arguments to override values in
                    `experiment_sample.input_data`.

                Returns
                -------
                ExperimentSample
                    The modified `ExperimentSample` instance with new stored
                    outputs.
                """
                # Extract input arguments from experiment sample
                _input = {name: experiment_sample.input_data.get(
                    name) for name in input_names if name not in kwargs}

                # Call the function
                _output = f(**_input, **kwargs)

                # Ensure the output is iterable
                if not isinstance(_output, tuple):
                    _output = (_output,)

                # Store outputs in the experiment sample
                for name, value in zip(output_names, _output):
                    experiment_sample.store(name=name, object=value)

                return experiment_sample

        return FunctionDataGenerator()

    return decorator


def create_datagenerator(
        data_generator: str, **parameters
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

        filtered_name = data_generator.lower().replace(
            ' ', '').replace('-', '').replace('_', '').replace('.', '')

        if filtered_name in DATAGENERATOR_MAPPING:
            return DATAGENERATOR_MAPPING[filtered_name](**parameters)

        else:
            raise KeyError(f"Unknown data generator name: {data_generator}")

    # If the data generator is not a known type, raise an error
    else:
        raise TypeError(f"Unknown data generator type: {type(data_generator)}")

# =============================================================================
