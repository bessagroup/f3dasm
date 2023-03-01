import inspect
from typing import List

from ..base.function import Function
from . import pybenchfunction
from .adapters.augmentor import *
from .pybenchfunction import *

_available_functions = inspect.getmembers(pybenchfunction, inspect.isclass)


def get_functions(
    d=None, continuous=None, convex=None, separable=None, differentiable=None, multimodal=None, randomized_term=None
) -> List[Function]:

    functions = [cls for clsname, cls in _available_functions if clsname not in [
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


FUNCTIONS = get_functions()
FUNCTIONS_2D = get_functions(d=2)
FUNCTIONS_7D = get_functions(d=7)
