#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
from copy import deepcopy
from enum import Enum
from pathlib import Path
from typing import Iterable, List, Type

# Third-party
import pandas as pd

# Local
from ._newdata2 import _Data

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

# =============================================================================


class NoOpenJobsError(Exception):
    """
    Exception raised when there are no open jobs.

    Attributes:
        message (str): The error message.
    """

    def __init__(self, message):
        super().__init__(message)

# =============================================================================


class Index:
    def __init__(self, jobs: pd.Series | None | str = None):
        """
        Initializes the Index object.

        Parameters
        ----------
        jobs : pd.Series, None, or str, optional
            Series of jobs, None, or a single job as a string.
        """
        if isinstance(jobs, str):
            self.jobs = pd.Series(jobs, index=[0], dtype='string')

        elif jobs is None:
            self.jobs = pd.Series(dtype='string')

        else:
            self.jobs = jobs

    def __len__(self) -> int:
        """
        Returns the number of jobs.

        Returns
        -------
        int
            Number of jobs.
        """
        return len(self.jobs)

    def __add__(self, __o: Index | str) -> Index:
        """
        Adds another Index or a string to this Index.

        Parameters
        ----------
        __o : Index or str
            Another Index object or a string representing a job.

        Returns
        -------
        Index
            A new Index object containing the combined jobs.
        """
        if isinstance(__o, str):
            __o = Index(__o)

        if self.jobs.empty:
            return __o

        # Make a copy of other.jobs and modify its index
        other_jobs_copy = deepcopy(__o)
        other_jobs_copy.jobs.index = pd.Index(
            range(len(other_jobs_copy))) + self.jobs.index[-1] + 1

        return Index(pd.concat([self.jobs, other_jobs_copy.jobs]))

    def __getitem__(self, indices: int | slice | Iterable[int]) -> Index:
        """
        Gets a subset of jobs by indices.

        Parameters
        ----------
        indices : int, slice, or Iterable[int]
            Indices to get.

        Returns
        -------
        Index
            A new Index object containing the selected jobs.
        """
        if isinstance(indices, int):
            indices = [indices]
        return Index(self.jobs[indices].copy())

    def __eq__(self, __o: Index) -> bool:
        """
        Checks if this Index is equal to another Index.

        Parameters
        ----------
        __o : Index
            Another Index object to compare.

        Returns
        -------
        bool
            True if the two Index objects are equal, False otherwise.
        """
        return self.jobs.equals(__o.jobs)

    def _repr_html_(self) -> str:
        """
        Returns an HTML representation of the jobs.

        Returns
        -------
        str
            HTML representation of the jobs.
        """
        return self.jobs.__repr__()

    @property
    def indices(self) -> pd.Index:
        """
        The indices of the jobs.

        Returns
        -------
        pd.Index
            The indices of the jobs.
        """
        return self.jobs.index

    def iloc(self, indices: Iterable[int] | int) -> Iterable[int]:
        """
        Gets the position of the given indices in the jobs.

        Parameters
        ----------
        indices : Iterable[int] or int
            Indices to locate.

        Returns
        -------
        Iterable[int]
            Positions of the given indices.
        """
        if isinstance(indices, int):
            indices = [indices]
        return self.indices.get_indexer(indices)

    def is_all_finished(self) -> bool:
        """
        Checks if all jobs are finished.

        Returns
        -------
        bool
            True if all jobs are finished, False otherwise.
        """
        return all(self.jobs.isin([Status.FINISHED, Status.ERROR]))

    @classmethod
    def from_data(cls: Type[Index], data: _Data,
                  value: str = Status.OPEN) -> Index:
        """
        Create an Index object from a Data object.

        Parameters
        ----------
        data : _Data
            Data object containing the data.
        value : str, optional
            The value to assign to the jobs. Can be 'open',
            'in_progress', 'finished', or 'error'. Default is 'open'.

        Returns
        -------
        Index
            Index object containing the loaded data.
        """
        return cls(pd.Series([value] * len(data), dtype='string'))

    @classmethod
    def from_file(cls: Type[Index], filename: Path | str) -> Index:
        """
        Create an Index object from a pickle file.

        Parameters
        ----------
        filename : Path or str
            Name of the file.

        Returns
        -------
        Index
            Index object containing the loaded data.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        """
        if Path(filename).with_suffix('.csv').exists():
            return cls(pd.read_csv(Path(filename).with_suffix('.csv'),
                                   index_col=0)['0'])
        elif Path(filename).with_suffix('.pkl').exists():
            return cls(pd.read_pickle(Path(filename).with_suffix('.pkl')))
        else:
            raise FileNotFoundError(f"Jobfile {filename} does not exist.")

    def select_all(self, status: str) -> Index:
        """
        Selects all jobs with a certain status.

        Parameters
        ----------
        status : str
            Status of the jobs to select.

        Returns
        -------
        Index
            Index object containing the selected jobs.
        """
        return Index(self.jobs[self.jobs == status])

    def store(self, filename: Path) -> None:
        """
        Stores the jobs in a pickle file.

        Parameters
        ----------
        filename : Path
            Path of the file.
        """
        self.jobs.to_pickle(filename.with_suffix('.pkl'))
        # self.jobs.to_csv(filename.with_suffix('.csv'))

    def to_dataframe(self, name: str = "") -> pd.DataFrame:
        """
        Converts the job queue to a DataFrame.

        Parameters
        ----------
        name : str, optional
            Name of the column. Default is an empty string.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the jobs.
        """
        return self.jobs.to_frame(name)

    def get_open_job(self) -> int:
        """
        Returns the index of an open job.

        Returns
        -------
        int
            Index of an open job.

        Raises
        ------
        NoOpenJobsError
            If no open jobs are found.
        """
        try:
            return int(self.jobs[self.jobs == Status.OPEN].index[0])
        except IndexError:
            raise NoOpenJobsError("No open jobs found.")

    def remove(self, indices: List[int]) -> None:
        """
        Removes a subset of the jobs.

        Parameters
        ----------
        indices : List[int]
            List of indices to remove.
        """
        self.jobs = self.jobs.drop(indices)

    def overwrite(self, indices: Iterable[int], other: Index | str) -> None:
        """
        Overwrites the jobs at the specified indices with new jobs.

        Parameters
        ----------
        indices : Iterable[int]
            Indices to overwrite.
        other : Index or str
            New jobs to overwrite with.
        """
        if isinstance(other, str):
            other = Index(pd.Series([other], index=[0], dtype='string'))

        self.jobs.update(other.jobs.set_axis(indices))

    def mark(self, index: int | slice | Iterable[int], status: Status) -> None:
        """
        Marks a job with a certain status.

        Parameters
        ----------
        index : int, slice, or Iterable[int]
            Index of the job to mark.
        status : Status
            Status to mark the job with.
        """
        self.jobs.loc[index] = status

    def mark_all_in_progress_open(self) -> None:
        """
        Marks all jobs as 'open'.
        """
        self.jobs = self.jobs.replace(Status.IN_PROGRESS, Status.OPEN)

    def mark_all_error_open(self) -> None:
        """
        Marks all jobs as 'open'.
        """
        self.jobs = self.jobs.replace(Status.ERROR, Status.OPEN)

    def reset_index(self) -> None:
        """
        Resets the index of the jobs.
        """
        self.jobs.reset_index(drop=True, inplace=True)


# =============================================================================

def _jobs_factory(jobs: Path | str | Index | None, input_data: _Data,
                  output_data: _Data, job_value: Status) -> Index:
    """Creates a Index object from particular inpute

    Parameters
    ----------
    jobs : Path | str | None
        input data for the jobs
    input_data : _Data
        _Data object of input data to extract indices from, if necessary
    output_data : _Data
        _Data object of output data to extract indices from, if necessary
    job_value : Status
        initial value of all the jobs

    Returns
    -------
    Index
        JobQueue object
    """
    if isinstance(jobs, Index):
        return jobs

    if isinstance(jobs, (Path, str)):
        return Index.from_file(Path(jobs))

    if input_data.is_empty():
        return Index.from_data(output_data, value=job_value)

    return Index.from_data(input_data, value=job_value)
