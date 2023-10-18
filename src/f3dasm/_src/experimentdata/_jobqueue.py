#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
from enum import Enum
from pathlib import Path
from typing import Iterable, List, Optional, Type

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


class Status(str, Enum):
    """Enum class for the status of a job."""
    OPEN = 'open'
    IN_PROGRESS = 'in progress'
    FINISHED = 'finished'
    ERROR = 'error'

    def __str__(self) -> str:
        return self.value


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

    def __add__(self, other: _JobQueue | int) -> _JobQueue:
        """Add two JobQueue objects together.

        Parameters
        ----------
        other : JobQueue
            JobQueue object to add.

        Returns
        -------
        JobQueue
            JobQueue object containing the added jobs.
        """
        if isinstance(other, int):
            # make _JobQueue from the jobnumber
            other = _JobQueue(pd.Series([Status.OPEN], index=[0], dtype='string'))

        try:
            last_index = self.jobs.index[-1]
        except IndexError:  # Empty Series
            return _JobQueue(other.jobs)

        # Make a copy of other.jobs and modify its index
        other_jobs_copy = other.jobs.copy()
        other_jobs_copy.index = other_jobs_copy.index + last_index + 1
        return _JobQueue(pd.concat([self.jobs, other_jobs_copy]))

    def __getitem__(self, index: int | slice | Iterable[int]) -> _Data:
        """Get a subset of the data.

        Parameters
        ----------
        index : int, slice, list
            The index of the data to get.

        Returns
        -------
            A subset of the data.
        """
        if isinstance(index, int):
            index = [index]
        return _JobQueue(self.jobs[index].copy())

    def __eq__(self, __o: _JobQueue) -> bool:
        return self.jobs.equals(__o.jobs)

    def _repr_html_(self) -> str:
        return self.jobs.__repr__()

    @property
    def indices(self) -> pd.Index:
        """The indices of the jobs."""
        return self.jobs.index
    #                                                      Alternative Constructors
    # =============================================================================

    @classmethod
    def from_data(cls: Type[_JobQueue], data: _Data, value: str = Status.OPEN) -> _JobQueue:
        """Create a JobQueue object from a Data object.

        Parameters
        ----------
        data : Data
            Data object containing the data.
        value : str
            The value to assign to the jobs. Can be 'open', 'in progress', 'finished', or 'error'.

        Returns
        -------
        JobQueue
            JobQueue object containing the loaded data.
        """
        return cls(pd.Series([value] * len(data), dtype='string'))

    @classmethod
    def from_file(cls: Type[_JobQueue], filename: Path | str) -> _JobQueue:
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
        # Convert filename to Path
        filename = Path(filename)

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

    def to_dataframe(self, name: str = "") -> pd.DataFrame:
        """Converts the job queue to a DataFrame.

        Parameters
        ----------
        name : str, optional
            Name of the column, by default "".

        Notes
        -----
        If the name is not specified, the column name will be an empty string

        Returns
        -------
        DataFrame
            DataFrame containing the jobs.
        """
        return self.jobs.to_frame("")

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

    def add(self, number_of_jobs: int = 1, status: str = Status.OPEN):
        """Adds a number of jobs to the job queue.

        Parameters
        ----------
        number_of_jobs : int, optional
            Number of jobs to add, by default 1
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

    #                                                                          Mark
    # =============================================================================

    def mark(self, index: int | slice | Iterable[int], status: Status) -> None:
        """Marks a job with a certain status.

        Parameters
        ----------
        index : int
            Index of the job to mark.
        status : str
            Status to mark the job with.
        """
        self.jobs.loc[index] = status

    def mark_all_in_progress_open(self) -> None:
        """Marks all jobs as 'open'."""
        self.jobs = self.jobs.replace(Status.IN_PROGRESS, Status.OPEN)

    def mark_all_error_open(self) -> None:
        """Marks all jobs as 'open'."""
        self.jobs = self.jobs.replace(Status.ERROR, Status.OPEN)
    #                                                                  Miscellanous
    # =============================================================================

    def is_all_finished(self) -> bool:
        """Checks if all jobs are finished.

        Returns
        -------
        bool
            True if all jobs are finished, False otherwise.
        """
        return all(self.jobs.isin([Status.FINISHED, Status.ERROR]))

    def get_open_job(self) -> int:
        """Returns the index of an open job.

        Returns
        -------
        int
            Index of an open job.
        """
        try:  # try to find an open job
            return int(self.jobs[self.jobs == Status.OPEN].index[0])
        except IndexError:
            raise NoOpenJobsError("No open jobs found.")

    def reset_index(self) -> None:
        """Resets the index of the jobs."""
        self.jobs.reset_index(drop=True, inplace=True)
