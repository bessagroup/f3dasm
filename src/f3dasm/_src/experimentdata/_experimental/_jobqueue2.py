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
        if isinstance(jobs, str):
            self.jobs = pd.Series(jobs, index=[0], dtype='string')

        elif jobs is None:
            self.jobs = pd.Series(dtype='string')

        else:
            self.jobs = jobs

    def __len__(self) -> int:
        return len(self.jobs)

    def __add__(self, __o: Index | str) -> Index:
        if isinstance(__o, str):
            __o = Index(__o)

        if self.jobs.empty:
            return __o

        # Make a copy of other.jobs and modify its index
        other_jobs_copy = deepcopy(__o)
        other_jobs_copy.jobs.index = range(
            len(other_jobs_copy)) + self.jobs.index[-1] + 1

        return Index(pd.concat([self.jobs, other_jobs_copy.jobs]))

    def __getitem__(self, indices: int | slice | Iterable[int]) -> Index:
        if isinstance(indices, int):
            indices = [indices]
        return Index(self.jobs[indices].copy())

    def __eq__(self, __o: Index) -> bool:
        return self.jobs.equals(__o.jobs)

    def _repr_html_(self) -> str:
        return self.jobs.__repr__()

    @property
    def indices(self) -> pd.Index:
        """The indices of the jobs."""
        return self.jobs.index

    def iloc(self, indices: Iterable[int]) -> Iterable[int]:
        return self.indices.get_indexer(indices)

    #                                                  Alternative Constructors
    # =========================================================================

    @classmethod
    def from_data(cls: Type[Index], data: _Data,
                  value: str = Status.OPEN) -> Index:
        """Create a JobQueue object from a Data object.

        Parameters
        ----------
        data : Data
            Data object containing the data.
        value : str
            The value to assign to the jobs. Can be 'open',
            'in progress', 'finished', or 'error'.

        Returns
        -------
        JobQueue
            JobQueue object containing the loaded data.
        """
        return cls(pd.Series([value] * len(data), dtype='string'))

    @classmethod
    def from_file(cls: Type[Index], filename: Path | str) -> Index:
        """Create a JobQueue object from a pickle file.

        Parameters
        ----------
        filename : Path | str
            Name of the file.

        Returns
        -------
        JobQueue
            JobQueue object containing the loaded data.
        """
        # Convert filename to Path
        if Path(filename).with_suffix('.csv').exists():
            return cls(
                pd.read_csv(Path(filename).with_suffix('.csv'),
                            index_col=0)['0'])

        elif Path(filename).with_suffix('.pkl').exists():
            return cls(
                pd.read_pickle(Path(filename).with_suffix('.pkl')))

        else:
            raise FileNotFoundError(f"Jobfile {filename} does not exist.")

    #                                                                    Select
    # =========================================================================

    def select_all(self, status: str) -> Index:
        """Selects all jobs with a certain status.

        Parameters
        ----------
        status : str
            Status of the jobs to select

        Returns
        -------
        JobQueue
            JobQueue object containing the selected jobs.
        """
        return Index(self.jobs[self.jobs == status])

    #                                                                    Export
    # =========================================================================

    def store(self, filename: Path) -> None:
        """Stores the jobs in a pickle file.

        Parameters
        ----------
        filename : Path
            Path of the file.
        """
        self.jobs.to_csv(filename.with_suffix('.csv'))

    def to_dataframe(self, name: str = "") -> pd.DataFrame:
        """Converts the job queue to a DataFrame.

        Parameters
        ----------
        name : str, optional
            Name of the column, by default "".

        Note
        ----
        If the name is not specified, the column name will be an empty string

        Returns
        -------
        DataFrame
            DataFrame containing the jobs.
        """
        return self.jobs.to_frame("")

    #                                                    Append and remove jobs
    # =========================================================================

    def remove(self, indices: List[int]):
        """Removes a subset of the jobs.

        Parameters
        ----------
        indices : List[int]
            List of indices to remove.
        """
        self.jobs = self.jobs.drop(indices)

    def overwrite(
            self, indices: Iterable[int],
            other: Index | str) -> None:

        if isinstance(other, str):
            other = Index(
                pd.Series([other], index=[0], dtype='string'))

        self.jobs.update(other.jobs.set_axis(indices))

    #                                                                      Mark
    # =========================================================================

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
    #                                                              Miscellanous
    # =========================================================================

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
