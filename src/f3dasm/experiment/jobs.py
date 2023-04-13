#                                                                       Modules
# =============================================================================

# Standard
import errno
import functools
import json
import logging
import os
from os import path
from time import sleep
from typing import Callable, Union

# import msvcrt if windows, otherwise (Unix system) import fcntl
if os.name == 'nt':
    import msvcrt
else:
    import fcntl

# Local
from ..design import ExperimentData

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class JobFileExistsError(Exception):
    """
    Exception raised when a job file already exists.

    Attributes:
        message (str): The error message.
    """

    def __init__(self, message):
        super().__init__(message)


class NoOpenJobsError(Exception):
    """
    Exception raised when there are no open jobs.

    Attributes:
        message (str): The error message.
    """

    def __init__(self, message):
        super().__init__(message)


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
                    # Try to open the jobs file
                    with open(f"{self.filename}.json", 'r+') as file:
                        if os.name == 'nt':
                            msvcrt.locking(file.fileno(), msvcrt.LK_LOCK, 1)
                        else:
                            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)

                        # Load the jobs data to the object
                        try:
                            self.create_jobs_from_dictionary(json.load(file))
                        except json.JSONDecodeError as e:
                            logging.exception(f"Failed to load JSON data from file {self.filename}.json")
                            raise e

                        # Do the operation
                        value = operation(self, *args, **kwargs)

                        # Delete existing contents of file
                        file.seek(0, 0)
                        file.truncate()

                        # Write the data back to the file
                        json.dump(self.jobs, file)

                    break
                except IOError as e:
                    # the file is locked by another process
                    if os.name == 'nt':
                        if e.errno == 13:
                            logging.info("The jobs file is currently locked by another process. "
                                         "Retrying in 1 second...")
                            sleep(sleeptime_sec)
                        elif e.errno == 2:  # File not found error
                            logging.info("The jobs file does not exist. Retrying in 1 second...")
                            sleep(sleeptime_sec)
                        else:
                            logging.info(f"An unexpected IOError occurred: {e}")
                            break
                    else:
                        if e.errno == errno.EAGAIN:
                            logging.info("The jobs file is currently locked by another process. "
                                         "Retrying in 1 second...")
                            sleep(sleeptime_sec)
                        elif e.errno == 2:  # File not found error
                            logging.info("The jobs file does not exist. Retrying in 1 second...")
                            sleep(sleeptime_sec)
                        else:
                            logging.info(f"An unexpected IOError occurred: {e}")
                            break
                except Exception as e:
                    # handle any other exceptions
                    logging.info(f"An unexpected error occurred: {e}")
                    raise e
                    return

            return value

        return wrapper_func

    return decorator_func


class JobQueue:
    def __init__(self, filename: str):
        """
        A class that represents a dictionary of jobs that can be marked as 'open', 'in progress',
        'finished', or 'error'.

        Parameters
        ----------
        filename : str
            The name of the file that the jobs will be saved in.
        """
        self.filename = filename

    def __repr__(self):
        return self.jobs.__repr__()

    def _set_value(self, index: int, value: str):
        self.jobs[index] = value

    def create_jobs_from_experimentdata(self, experimentdata: ExperimentData):
        """Create jobs based on an ExperimentData object.

        Parameters
        ----------
        experimentdata : ExperimentData
            The ExperimentData object that will be used to create the jobs.
        """
        self.jobs = {index: 'open' for index in range(experimentdata.get_number_of_datapoints())}

    def create_jobs_from_dictionary(self, dictionary: dict):
        """Create jobs based on a dictionary.

        Parameters
        ----------
        dictionary : dict
            The dictionary that will be used to create the jobs.

        """
        new_dict = {}
        for k, v in dictionary.items():
            new_dict[int(k)] = v

        self.jobs = new_dict

    @access_file()
    def get_jobs(self) -> dict:
        """Get the jobs as a dictionary.

        Returns
        -------
        jobs : dict
            A dictionary of jobs, where the keys are integers and the values are strings
            representing the status of the job.
        """
        return self.jobs

    @access_file()
    def mark_finished(self, index: int):
        """Mark a job as 'finished'.

        Parameters
        ----------
        index : int
            The index of the job to be marked as 'finished'.
        """
        self._set_value(index, 'finished')

    @access_file()
    def mark_error(self, index: int):
        """Mark a job as 'error'.

        Parameters
        ----------
        index : int
            The index of the job to be marked as 'error'.
        """
        self._set_value(index, 'error')

    @access_file()
    def add_job(self):
        """Add a new job to the list."""
        end = len(self.jobs)
        self._set_value(end, 'open')

    @access_file()
    def get(self) -> int:
        """Get the index of an 'open' job.

        Returns
        -------
        index : int or None
            The index of an 'open' job, or None if no 'open' job is present.
        """
        for key, value in self.jobs.items():
            if value == 'open':
                self.jobs[key] = 'in progress'
                return key

        raise NoOpenJobsError(f"The jobfile {self.filename} does not have any open jobs left!")

    def write_new_jobfile(self):
        filename = f"{self.filename}.json"
        if path.exists(filename):
            raise JobFileExistsError(f"The jobfile {filename} already exists!")

        with open(filename, 'w') as f:
            json.dump(self.jobs, f)
