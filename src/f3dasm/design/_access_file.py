import errno
import functools
import os
from pathlib import Path
from time import sleep
from typing import Callable

from filelock import FileLock

from ..logger import logger
from ._data import _Data
from ._jobqueue import _JobQueue

# from .experimentdata import ExperimentData


# import msvcrt if windows, otherwise (Unix system) import fcntl
if os.name == 'nt':
    import msvcrt
else:
    import fcntl


# def access_file(sleeptime_sec: int = 1) -> Callable:
#     """Wrapper for accessing a single resource with a file lock

#     Parameters
#     ----------
#     sleeptime_sec, optional
#         number of seconds to wait before trying to access resource again, by default 1

#     Returns
#     -------
#     decorator
#     """
#     def decorator_func(operation: Callable) -> Callable:
#         @functools.wraps(operation)
#         def wrapper_func(self, *args, **kwargs) -> None:
#             lock = FileLock(f"{self.filename}.lock")
#             with lock:
#                 self = ExperimentData.from_file(filename=Path(self.filename))
#                 value = operation(self, *args, **kwargs)
#                 self.store(filename=Path(self.filename))
#             return value

#         return wrapper_func

#     return decorator_func
