"""
Module for analytical benchmark functions
"""

#                                                                       Modules
# =============================================================================

# Standard
import inspect
from typing import List, Optional, Type

# Local
from . import pybenchfunction
from .function import Function

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


_available_functions = inspect.getmembers(pybenchfunction, inspect.isclass)


def get_function_classes(
    d: Optional[int] = None, continuous: Optional[bool] = None,
    convex: Optional[bool] = None,
    separable: Optional[bool] = None, differentiable: Optional[bool] = None,
    multimodal: Optional[bool] = None, randomized_term: Optional[bool] = None
) -> List[Type[Function]]:
    """Get a list of benchmark functions based on the given parameters

    Parameters
    ----------
    d : Optional[int], optional
        number of dimensions, by default None
    continuous : Optional[bool], optional
        filter for continuous functions, by default None
    convex : Optional[bool], optional
        filter for convex functions, by default None
    separable : Optional[bool], optional
        filter for separable functions, by default None
    differentiable : Optional[bool], optional
        filter for differentiable functions, by default None
    multimodal : Optional[bool], optional
        filter for multimodal functions, by default None
    randomized_term : Optional[bool], optional
        filter for functions that have a randomized term, by default None

    Returns
    -------
    List[Function]
        List of function classes that match the given parameters
    """

    functions = [cls for clsname, cls in _available_functions
                 if clsname not in [
                     "Function", "PyBenchFunction"]]

    functions = list(filter(lambda f: (d is None) or (
        f.is_dim_compatible(d)), functions))
    functions = list(filter(lambda f: (continuous is None)
                     or (f.continuous == continuous), functions))
    functions = list(filter(lambda f: (convex is None)
                     or (f.convex == convex), functions))
    functions = list(filter(lambda f: (separable is None)
                     or (f.separable == separable), functions))
    functions = list(filter(lambda f: (differentiable is None)
                     or (f.differentiable == differentiable), functions))
    functions = list(filter(lambda f: (multimodal is None)
                     or (f.multimodal == multimodal), functions))
    functions = list(filter(lambda f: (randomized_term is None) or (
        f.randomized_term == randomized_term), functions))

    return functions


def get_functions(
    d: Optional[int] = None, continuous: Optional[bool] = None,
    convex: Optional[bool] = None,
    separable: Optional[bool] = None, differentiable: Optional[bool] = None,
    multimodal: Optional[bool] = None, randomized_term: Optional[bool] = None
) -> List[str]:
    """Get a list of benchmark functions based on the given parameters

    Parameters
    ----------
    d : Optional[int], optional
        number of dimensions, by default None
    continuous : Optional[bool], optional
        filter for continuous functions, by default None
    convex : Optional[bool], optional
        filter for convex functions, by default None
    separable : Optional[bool], optional
        filter for separable functions, by default None
    differentiable : Optional[bool], optional
        filter for differentiable functions, by default None
    multimodal : Optional[bool], optional
        filter for multimodal functions, by default None
    randomized_term : Optional[bool], optional
        filter for functions that have a randomized term, by default None

    Returns
    -------
    List[Function]
        List of function classes that match the given parameters
    """
    function_classes = get_function_classes(
        d, continuous, convex, separable,
        differentiable, multimodal, randomized_term)
    return [function_class.name for function_class in function_classes]


_DATAGENERATORS = get_function_classes()
_FUNCTIONS_2D = get_function_classes(d=2)
_FUNCTIONS_7D = get_function_classes(d=7)

FUNCTIONS = get_functions()
FUNCTIONS_2D = get_functions(d=2)
FUNCTIONS_7D = get_functions(d=7)


def find_function(query: str) -> Function:
    """Find a function from the f3dasm.functions submodule
    Parameters
    ----------
    query
        string representation of the requested function
    Returns
    -------
        class of the requested function
    """
    try:
        return list(
            filter(lambda function: function.__name__ == query,
                   _DATAGENERATORS))[0]
    except IndexError:
        return ValueError(f'Function {query} not found!')
