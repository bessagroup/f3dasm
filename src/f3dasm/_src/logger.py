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
from pathlib import Path
from time import sleep
from typing import TextIO
import yaml

if os.name == 'nt':
    import msvcrt
else:
    import fcntl

from functools import partial, wraps
from time import perf_counter
from typing import Any, Callable

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


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

                break
            except IOError as e:
                # the file is locked by another process
                if e.errno == errno.EAGAIN:
                    logger.debug(
                        "The log file is currently locked by another process. Retrying in 1 second..."
                    )
                    sleep(1)
                else:
                    logger.info(f"An unexpected IOError occurred: {e}")
                    break
            except Exception as e:
                # handle any other exceptions
                logger.info(f"An unexpected error occurred: {e}")
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


def _time_and_log(func: Callable, logger=logging.Logger) -> Callable:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = perf_counter()
        value = func(*args, **kwargs)
        logger.debug(
            f"Called {func.__name__} and time taken: {perf_counter() - start_time:.2f}s"
        )
        return value

    return wrapper

# f3dasm can be used as a library or as an application
# If used as a library, the user is responsible to define a f3dasm logger
# Otherwise, the default behavior is to load the f3dsam logging configuration

logger_name = "f3dsam"

# Check for the existence of the f3dsam logger
# This is required because getLogger() creates the logger if it does not exist
# Loggers are stored in the logger manager (undocumented feature)
if logger_name not in logging.Logger.manager.loggerDict.keys():
    # If not, load the logging configuration that should create a f3dasm logger
    config_path = Path(__file__).parent/"logger_config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)

# Get existing logger without overwriting the existing configuration
logger = logging.getLogger(name="f3dsam")

time_and_log = partial(_time_and_log, logger=logger)
