from ..logger import logger
from typing import Callable
import functools
import os
from time import sleep
import errno

from ._data import _Data
from ._jobqueue import _JobQueue

# import msvcrt if windows, otherwise (Unix system) import fcntl
if os.name == 'nt':
    import msvcrt
else:
    import fcntl

def access_file(sleeptime_sec: int = 1) -> Callable:
    """Wrapper for accessing a single resource with a file lock

    Parameters
    ----------
    sleeptime_sec, optional
        number of seconds to wait before trying to access resource again, by default 1

    Returns
    -------
    decorator
    """
    def decorator_func(operation: Callable) -> Callable:
        @functools.wraps(operation)
        def wrapper_func(self, *args, **kwargs) -> None:
            while True:
                try:
                    # Try to open the experimentdata file
                    logger.debug(f"Trying to open the data file: {self.filename}_data.csv")
                    with open(f"{self.filename}_data.csv", 'rb+') as file:
                        logger.debug("Opened file successfully")
                        if os.name == 'nt':
                            msvcrt.locking(file.fileno(), msvcrt.LK_LOCK, 1)
                        else:
                            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            logger.debug("Locked file successfully")

                        # Load the experimentdata from the object
                        self.data = _Data.from_file(filename=self.filename, text_io=file)
                        logger.debug("Loaded data successfully")

                        # Load the jobs from disk
                        self.jobs = _JobQueue.from_file(filename=f"{self.filename}_jobs")
                        logger.debug("Loaded jobs successfully")

                        logger.debug(
                            f"Executing operation {operation.__name__} with args: {args} and kwargs: {kwargs}")
                        # Do the operation
                        value = operation(self, *args, **kwargs)

                        logger.debug("Executed operation succesfully")
                        # Delete existing contents of file
                        file.seek(0, 0)
                        file.truncate()

                        # Write the data to disk
                        self.data.store(filename=f"{self.filename}_data", text_io=file)
                        self.jobs.store(filename=f"{self.filename}_jobs")

                    break
                except IOError as e:
                    # the file is locked by another process
                    if os.name == 'nt':
                        if e.errno == 13:
                            logger.info("The data file is currently locked by another process. "
                                        "Retrying in 1 second...")
                            sleep(sleeptime_sec)
                        elif e.errno == 2:  # File not found error
                            logger.info("The data file does not exist. Retrying in 1 second...")
                            sleep(sleeptime_sec)
                        else:
                            logger.info(f"An unexpected IOError occurred: {e}")
                            break
                    else:
                        if e.errno == errno.EAGAIN:
                            logger.info("The data file is currently locked by another process. "
                                        "Retrying in 1 second...")
                            sleep(sleeptime_sec)
                        elif e.errno == 2:  # File not found error
                            logger.info("The data file does not exist. Retrying in 1 second...")
                            sleep(sleeptime_sec)
                        else:
                            logger.info(f"An unexpected IOError occurred: {e}")
                            break
                except Exception as e:
                    # handle any other exceptions
                    logger.info(f"An unexpected error occurred: {e}")
                    raise e
                    return

            return value

        return wrapper_func

    return decorator_func
