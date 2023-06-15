"""
This module defines the logging settings for the f3dasm package.
"""
#                                                                       Modules
# =============================================================================

# Standard
import errno
import logging
import os
from logging import FileHandler, StreamHandler
from time import sleep
from typing import TextIO

if os.name == 'nt':
    import msvcrt
else:
    import fcntl

from functools import partial, wraps
from time import perf_counter
from typing import Any, Callable

# Local
from ._show_versions import __version__

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

# Logging things
logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
logging.getLogger('tensorflow').setLevel(logging.ERROR)


class DistributedFileHandler(FileHandler):
    def __init__(self, filename):
        """Distributed FileHandler class for handling logging to
        one single file when multiple nodes access the same resource

        Parameters
        ----------
        filename
            name of the logging file
        """
        super().__init__(filename)

    def emit(self, record):
        """
        Emit a record.

        If the stream was not opened because 'delay' was specified in the
        constructor, open it before calling the superclass's emit.
        """
        while True:
            try:
                if self.stream is None:
                    self.stream = self._open()

                _lock_file(self.stream)
                StreamHandler.emit(self, record)

                _unlock_file(self.stream)
                # print("Succesfully logged!")

                break
            except IOError as e:
                # the file is locked by another process
                if e.errno == errno.EAGAIN:
                    print("The log file is currently locked by another process. Retrying in 1 second...")
                    sleep(1)
                else:
                    print(f"An unexpected IOError occurred: {e}")
                    break
            except Exception as e:
                # handle any other exceptions
                print(f"An unexpected error occurred: {e}")
                break


def _lock_file(file):
    """Lock the file with the lock

    Parameters
    ----------
    file
        file object that returns from open()
    """
    if os.name == 'nt':  # for Windows
        msvcrt.locking(file.fileno(), msvcrt.LK_LOCK, 1)
    else:  # for Unix
        fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock_file(file):
    """Unlock the file with the lock

    Parameters
    ----------
    file
        file object that returns from open()
    """
    if os.name == 'nt':  # for Windows
        msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
    else:  # for Unix
        fcntl.flock(file, fcntl.LOCK_UN)


def create_logger(name: str, level: int = logging.INFO, filename: str = None) -> logging.Logger:
    """Create a logger

    Parameters
    ----------
    name : str
        Name of the logger
    level : int, optional
        Logging level, by default logging.INFO
    filename : str, optional
        Filename of the log file, by default None

    Returns
    -------
    logging.Logger
        The created logger
    """

    # Create a logger
    logger = logging.getLogger(name)

    # Set the logging level
    logger.setLevel(level)

    # Create a file handler
    if filename is None:
        filename = f"{name}.log"

    # Check if file ends with .log
    if not filename.endswith(".log"):
        filename += ".log"

    handler = DistributedFileHandler(filename)

    # Create a logging format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(handler)

    return logger


def _time_and_log(
    func: Callable, logger=logging.Logger
) -> Callable:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = perf_counter()
        value = func(*args, **kwargs)
        logger.info(f"Called {func.__name__} and time taken: {perf_counter() - start_time:.2f}s")
        return value

    return wrapper


# Create a logger
logger = logging.getLogger("f3dasm")
time_and_log = partial(_time_and_log, logger=logger)

# Log welcome message and the version of f3dasm
logger.info(f"Imported f3dasm (version: {__version__})")
