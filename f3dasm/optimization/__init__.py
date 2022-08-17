"""
Some API information about the opitmizers
"""


import inspect
from . import (
    gpyopt_implementations,
    gradient_based_algorithms,
    pygmo_implementations,
    scipy_implementations,
    randomsearch,
)

from .gpyopt_implementations import *
from .gradient_based_algorithms import *
from .pygmo_implementations import *
from .scipy_implementations import *

_available_optimizers_gpyopt_implementations = inspect.getmembers(gpyopt_implementations, inspect.isclass)
_available_optimizers_gradient_based_algorithms = inspect.getmembers(gradient_based_algorithms, inspect.isclass)
_available_optimizers_pygmo_implementations = inspect.getmembers(pygmo_implementations, inspect.isclass)
_available_optimizers_scipy_implementations = inspect.getmembers(scipy_implementations, inspect.isclass)
_available_optimizers_random_search = inspect.getmembers(randomsearch, inspect.isclass)


_available_optimizers = (
    _available_optimizers_gpyopt_implementations
    + _available_optimizers_gradient_based_algorithms
    + _available_optimizers_pygmo_implementations
    + _available_optimizers_scipy_implementations
    + _available_optimizers_random_search
)

OPTIMIZERS = [
    cls
    for clsname, cls in _available_optimizers
    if clsname
    not in [
        "Optimizer",
        "DesignSpace",
        "Function",
        "PygmoProblem",
        "PygmoAlgorithm",
        "SciPyMinimizeOptimizer",
        "SciPyOptimizer",
        "BayesianOptimization",
    ]  # Discard BO because slow for testing
]
