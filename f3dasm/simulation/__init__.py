import inspect
from . import pybenchfunction

_available_functions = inspect.getmembers(pybenchfunction, inspect.isclass)

FUNCTIONS = [
    cls
    for clsname, cls in _available_functions
    if clsname not in ["Function", "PyBenchFunction"]
]

print("test")
