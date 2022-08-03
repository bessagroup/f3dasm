import inspect
from . import (
    gpyopt_implementations,
    gradient_based_algorithms,
    pygmo_implementations,
    scipy_implementations,
)

_available_optimizers_gpyopt_implementations = inspect.getmembers(
    gpyopt_implementations, inspect.isclass
)
_available_optimizers_gradient_based_algorithms = inspect.getmembers(
    gradient_based_algorithms, inspect.isclass
)
_available_optimizers_pygmo_implementations = inspect.getmembers(
    pygmo_implementations, inspect.isclass
)

_available_optimizers_scipy_implementations = inspect.getmembers(
    scipy_implementations, inspect.isclass
)

_available_optimizers = (
    _available_optimizers_gpyopt_implementations
    + _available_optimizers_gradient_based_algorithms
    + _available_optimizers_pygmo_implementations
    + _available_optimizers_scipy_implementations
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
        "SciPyLocalOptimizer",
        "SciPyGlobalOptimizer",
        "BayesianOptimization",
    ]  # Discard BO because slow for testing
]
