#                                                                       Modules
# =============================================================================


# Standard
import functools
from typing import Any, Callable, Dict, Iterator, List, Protocol, Tuple

# Third-party core
from pathos.helpers import mp

# Local
from .._logging import logger
from ..design._jobqueue import NoOpenJobsError

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class Trial(Protocol):
    @property
    def _jobnumber(self) -> int:
        ...

    @property
    def _dict_input(self) -> Dict[str, Any]:
        ...

    @property
    def _dict_output(self) -> Dict[str, Any]:
        ...


class ExperimentData(Protocol):

    @property
    def filename(self) -> str:
        ...

    def __iter__(self) -> Iterator[Tuple[Dict[str, Any]]]:
        ...

    def access_open_job_data(self) -> Trial:
        ...

    def write_error(self, job_id: int) -> None:
        ...

    def set_error(self, trial: Trial) -> None:
        ...

    def store(self) -> None:
        ...

    def get_open_job_data(self) -> Trial:
        ...

    def set_trial(self, trial: Trial) -> None:
        ...

    def write_trial(self, trial: Trial) -> None:
        ...

    @classmethod
    def from_file(cls, filename: str) -> 'ExperimentData':
        ...


def run_operation_on_experiments(data: ExperimentData, operation: Callable,
                                 parallel: bool = False, **kwargs) -> List[Any]:
    """Run an operation on a list of experiments

    Parameters
    ----------
    data
        ExperimtentData object
    operation
        Callable function that accepts one ExperimentData line
    parallel, optional
        Flag if the operation should be done in parallel, by default False

    Returns
    -------
        List of returns from the operation
    """

    if parallel:
        options = [
            ({'index': index, 'value_input': value_input, 'value_output': value_output, **kwargs},)
            for index, (value_input, value_output) in enumerate(data)
        ]

        def f(options: Dict[str, Any]) -> Any:
            """This function wraps the operation to unpack the options dictionary"""
            return operation(**options)

        with mp.Pool() as pool:
            # maybe implement pool.starmap_async ?
            results = pool.starmap(f, options)

    else:
        results = []
        for index, (value_input, value_output) in enumerate(data):
            results.append(operation(index, value_input, value_output, **kwargs))

    return results


def run_on_experimentdata(data: ExperimentData, mode: str = "sequential"):
    def decorator_func(operation: Callable) -> Callable:
        if mode == "sequential":
            return _sequential_decorator(data)(operation)
        elif mode == "parallel":
            return _multiprocessing_decorator(data)(operation)
        elif mode == "cluster":
            return cluster_decorator(data)(operation)
        else:
            raise ValueError("Invalid paralleliation mode specified.")

    return decorator_func


def _sequential_decorator(data: ExperimentData):
    def decorator_func(operation: Callable) -> Callable:
        @functools.wraps(operation)
        def wrapper_func(*args, **kwargs) -> ExperimentData:
            while True:
                try:
                    trial = data.access_open_job_data()
                except NoOpenJobsError:
                    break

                try:
                    _trial = operation(trial, **kwargs)  # no *args!
                except Exception:
                    data.set_error(trial._jobnumber)

                data.set_trial(_trial)

            return data
        wrapper_func.original_function = operation  # Add attribute to store the original unwrapped function
        return wrapper_func
    return decorator_func


def _multiprocessing_decorator(data: ExperimentData):
    def decorator_func(operation: Callable) -> Callable:
        @functools.wraps(operation)
        def wrapper_func(*args, **kwargs) -> ExperimentData:

            # Get all the jobs
            options = []
            while True:
                try:
                    trial = data.access_open_job_data()
                    options.append(
                        ({'trial': trial, **kwargs},))
                except NoOpenJobsError:
                    break

                def f(options: Dict[str, Any]) -> Any:
                    return operation(**options)

                with mp.Pool() as pool:
                    # maybe implement pool.starmap_async ?
                    _trials: List[Trial] = pool.starmap(f, options)

                for _trial in _trials:
                    data.set_trial(_trial)

            return data
        wrapper_func.original_function = operation  # Add attribute to store the original unwrapped function
        return wrapper_func
    return decorator_func


def cluster_decorator(data: ExperimentData):
    def decorator_func(operation: Callable) -> Callable:
        @functools.wraps(operation)
        def wrapper_func(*args, **kwargs) -> ExperimentData:

            # Retrieve the updated experimentdata object from disc
            try:
                _data = data.from_file(data.filename)
            except FileNotFoundError:  # If not found, store current
                data.store()
                _data = data.from_file(data.filename)

            while True:
                try:
                    trial = _data.get_open_job_data()
                except NoOpenJobsError:
                    break

                try:
                    _trial = operation(trial, **kwargs)
                except Exception:
                    _data.write_error(_trial._jobnumber)

                _data.write_trial(trial)

            return _data
        wrapper_func.original_function = operation  # Add attribute to store the original unwrapped function
        return wrapper_func
    return decorator_func
