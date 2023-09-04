#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
from pathlib import Path
from typing import List, Optional, Type

# Third-party
import pandas as pd

# Local
from ._data import _Data

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

OPEN = 'open'
IN_PROGRESS = 'in progress'
FINISHED = 'finished'
ERROR = 'error'


class NoOpenJobsError(Exception):
    """
    Exception raised when there are no open jobs.

    Attributes:
        message (str): The error message.
    """

    def __init__(self, message):
        super().__init__(message)


class _JobQueue:
    def __init__(self, jobs: Optional[pd.Series] = None):
        """
        A class that represents a dictionary of jobs that can be marked as 'open', 'in progress',
        'finished', or 'error'.

        Parameters
        ----------
        filename : str
            The name of the file that the jobs will be saved in.
        """
        if jobs is None:
            jobs = pd.Series(dtype='string')

        self.jobs: pd.Series = jobs

    def _repr_html_(self) -> str:
        return self.jobs.__repr__()

    #                                                      Alternative Constructors
    # =============================================================================

    @classmethod
    def from_data(cls: Type[_JobQueue], data: _Data):
        """Create a JobQueue object from a Data object.

        Parameters
        ----------
        data : Data
            Data object containing the data.

        Returns
        -------
        JobQueue
            JobQueue object containing the loaded data.
        """
        return cls(pd.Series([OPEN] * len(data), dtype='string'))

    @classmethod
    def from_file(cls: Type[_JobQueue], filename: Path) -> _JobQueue:
        """Create a JobQueue object from a pickle file.

        Parameters
        ----------
        filename : str
            Name of the file.

        Returns
        -------
        JobQueue
            JobQueue object containing the loaded data.
        """
        # Check if the file exists
        if not filename.with_suffix('.pkl').exists():
            raise FileNotFoundError(f"Jobfile {filename} does not exist.")

        return cls(pd.read_pickle(filename.with_suffix('.pkl')))

    def reset(self) -> None:
        """Resets the job queue."""
        self.jobs = pd.Series(dtype='string')

    #                                                                        Export
    # =============================================================================

    def store(self, filename: Path) -> None:
        """Stores the jobs in a pickle file.

        Parameters
        ----------
        filename : Path
            Path of the file.
        """
        self.jobs.to_pickle(filename.with_suffix('.pkl'))

    #                                                        Append and remove jobs
    # =============================================================================

    def remove(self, indices: List[int]):
        """Removes a subset of the jobs.

        Parameters
        ----------
        indices : List[int]
            List of indices to remove.
        """
        self.jobs = self.jobs.drop(indices)

    def add(self, number_of_jobs: int, status: str = OPEN):
        """Adds a number of jobs to the job queue.

        Parameters
        ----------
        number_of_jobs : int
            Number of jobs to add.
        status : str, optional
            Status of the jobs, by default 'open'.
        """
        try:
            last_index = self.jobs.index[-1]
        except IndexError:  # Empty Series
            self.jobs = pd.Series([status] * number_of_jobs, dtype='string')
            return

        new_indices = pd.RangeIndex(start=last_index + 1, stop=last_index + number_of_jobs + 1, step=1)
        jobs_to_add = pd.Series(status, index=new_indices, dtype='string')
        self.jobs = pd.concat([self.jobs, jobs_to_add], ignore_index=False)

    def select(self, indices: List[int]):
        """Selects a subset of the jobs.

        Parameters
        ----------
        indices : List[int]
            List of indices to select.
        """
        self.jobs = self.jobs.loc[indices]

    #                                                                          Mark
    # =============================================================================

    def mark_as_in_progress(self, index: int) -> None:
        """Marks a job as in progress.

        Parameters
        ----------
        index : int
            Index of the job to mark as in progress.
        """
        self.jobs.loc[index] = IN_PROGRESS

    def mark_as_finished(self, index: int) -> None:
        """Marks a job as finished.

        Parameters
        ----------
        index : int
            Index of the job to mark as finished.
        """
        self.jobs.loc[index] = FINISHED

    def mark_as_error(self, index: int) -> None:
        """Marks a job as finished.

        Parameters
        ----------
        index : int
            Index of the job to mark as finished.
        """
        self.jobs.loc[index] = ERROR

    def mark_all_in_progress_open(self) -> None:
        """Marks all jobs as 'open'."""
        self.jobs = self.jobs.replace(IN_PROGRESS, OPEN)

    def mark_all_open(self) -> None:
        """Marks all jobs as 'open'."""
        self.jobs = self.jobs.replace([IN_PROGRESS, FINISHED, ERROR], OPEN)

    #                                                                  Miscellanous
    # =============================================================================

    def is_all_finished(self) -> bool:
        """Checks if all jobs are finished.

        Returns
        -------
        bool
            True if all jobs are finished, False otherwise.
        """
        return all(self.jobs.isin([FINISHED, ERROR]))

    def get_open_job(self) -> int:
        """Returns the index of an open job.

        Returns
        -------
        int
            Index of an open job.
        """
        try:  # try to find an open job
            return int(self.jobs[self.jobs == OPEN].index[0])
        except IndexError:
            raise NoOpenJobsError("No open jobs found.")
