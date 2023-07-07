#                                                                       Modules
# =============================================================================

# Standard
from typing import List, Union

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


class NoOpenJobsError(Exception):
    """
    Exception raised when there are no open jobs.

    Attributes:
        message (str): The error message.
    """

    def __init__(self, message):
        super().__init__(message)


class _JobQueue:
    def __init__(self, jobs: pd.Series = None):
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

    @classmethod
    def from_data(cls, data: _Data):
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
        return cls(pd.Series(['open'] * data.number_of_datapoints(), dtype='string'))

    @classmethod
    def from_file(cls, filename: str) -> '_JobQueue':
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

        # if filename does not end with .pkl, add it
        if not filename.endswith('.pkl'):
            filename = filename + '.pkl'

        return cls(pd.read_pickle(filename))

    def store(self, filename: str) -> None:
        """Stores the jobs in a pickle file.

        Parameters
        ----------
        filename : str
            Name of the file.
        """

        # if filename does not end with .pkl, add it
        if not filename.endswith('.pkl'):
            filename = filename + '.pkl'

        self.jobs.to_pickle(filename)

    def select(self, indices: List[int]):
        """Selects a subset of the jobs.

        Parameters
        ----------
        indices : List[int]
            List of indices to select.
        """
        self.jobs = self.jobs.loc[indices]

    def remove(self, indices: List[int]):
        """Removes a subset of the jobs.

        Parameters
        ----------
        indices : List[int]
            List of indices to remove.
        """
        self.jobs = self.jobs.drop(indices)

    def add(self, number_of_jobs: int, status: str = 'open'):
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

    def reset(self) -> None:
        """Resets the job queue."""
        self.jobs = pd.Series(dtype='string')

    def get_open_job(self) -> Union[int, None]:
        """Returns the index of an open job.

        Returns
        -------
        int
            Index of an open job.
        """
        try:  # try to find an open job
            return int(self.jobs[self.jobs == 'open'].index[0])
        except IndexError:
            raise NoOpenJobsError("No open jobs found.")

    def mark_as_in_progress(self, index: int) -> None:
        """Marks a job as in progress.

        Parameters
        ----------
        index : int
            Index of the job to mark as in progress.
        """
        self.jobs.loc[index] = 'in progress'

    def mark_as_finished(self, index: int) -> None:
        """Marks a job as finished.

        Parameters
        ----------
        index : int
            Index of the job to mark as finished.
        """
        self.jobs.loc[index] = 'finished'

    def mark_as_error(self, index: int) -> None:
        """Marks a job as finished.

        Parameters
        ----------
        index : int
            Index of the job to mark as finished.
        """
        self.jobs.loc[index] = 'error'

    def mark_all_in_progress_open(self) -> None:
        """Marks all jobs as 'open'."""
        self.jobs = self.jobs.replace('in progress', 'open')

    def mark_all_open(self) -> None:
        """Marks all jobs as 'open'."""
        self.jobs = self.jobs.replace(['in progress', 'finished', 'error'], 'open')

    def is_all_finished(self) -> bool:
        """Checks if all jobs are finished.

        Returns
        -------
        bool
            True if all jobs are finished, False otherwise.
        """
        return all(self.jobs.isin(['finished', 'error']))
