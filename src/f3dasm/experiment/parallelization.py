#                                                                       Modules
# =============================================================================


# Standard
import functools
from typing import Any, Callable, Dict, Iterator, List, Protocol, Tuple

# Third-party core
from pathos.helpers import mp

# Local
from ..design._jobqueue import NoOpenJobsError

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


# class NoOpenJobsError(Exception):
#     ...


class ExperimentData(Protocol):
    def __iter__(self) -> Iterator[Tuple[Dict[str, Any]]]:
        ...

    def access_open_job_data(self) -> Tuple[int, Dict[str, Any], Dict[str, Any]]:
        ...

    def set_outputdata_by_index(self, index: int, value: Any) -> None:
        ...

    def write_error(self, job_id: int) -> None:
        ...

    def store(self) -> None:
        ...

    def write_outputdata_by_index(self, index: int, value: Any) -> None:
        ...

    def get_open_job_data(self) -> Tuple[int, Dict[str, Any], Dict[str, Any]]:
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
                    job_id, value_input, _ = data.access_open_job_data()
                except NoOpenJobsError:
                    break

                try:
                    result = operation(value_input, **kwargs)  # no *args!
                except Exception:
                    data.write_error(job_id)

                data.set_outputdata_by_index(job_id, result)

            return data
        return wrapper_func
    return decorator_func


def _multiprocessing_decorator(data: ExperimentData):
    def decorator_func(operation: Callable) -> Callable:
        @functools.wraps(operation)
        def wrapper_func(*args, **kwargs) -> ExperimentData:

            # Get all the jobs
            options, job_ids = [], []
            while True:
                try:
                    job_id, value_input, _ = data.access_open_job_data()
                    options.append(
                        ({'value_input': value_input, **kwargs},))
                    job_ids.append(job_id)
                except NoOpenJobsError:
                    break

                def f(options: Dict[str, Any]) -> Any:
                    return operation(**options)

                with mp.Pool() as pool:
                    # maybe implement pool.starmap_async ?
                    results = pool.starmap(f, options)

                for index, result in enumerate(results):
                    data.set_outputdata_by_index(
                        job_ids[index], result)

            return data
        return wrapper_func
    return decorator_func


def cluster_decorator(data: ExperimentData):
    def decorator_func(operation: Callable) -> Callable:
        @functools.wraps(operation)
        def wrapper_func(*args, **kwargs) -> ExperimentData:

            # Retrieve the updated experimentdata object from disc
            try:
                _data = ExperimentData.from_file(data.filename)
            except FileNotFoundError:  # If not found, store current
                data.store()
                _data = ExperimentData.from_file(data.filename)

            while True:
                try:
                    job_id, value_input, _ = _data.get_open_job_data()
                except NoOpenJobsError:
                    break

                try:
                    result = operation(value_input, **kwargs)
                except Exception:
                    _data.write_error(job_id)

                _data.write_outputdata_by_index(job_id, result)

            return _data
        return wrapper_func
    return decorator_func
