"""
This module contains the DataGenerator abstraction.
"""
#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
import traceback
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

# Third-party
from filelock import FileLock
from pathos.helpers import mp

# Local
from ._io import EXPERIMENTDATA_SUBFOLDER, LOCK_FILENAME, MAX_TRIES
from .experimentdata import ExperimentData
from .experimentsample import ExperimentSample
from .logger import logger
from .mpi_utils import (
    mpi_get_open_job,
    mpi_lock_manager,
    mpi_store_experiment_sample,
    mpi_terminate_worker,
)

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Alpha'
# =============================================================================
#
# =============================================================================

# =========================================================================


def _evaluate_sequential(execute_fn: Callable, data: ExperimentData,
                         pass_id: bool,
                         **kwargs) -> ExperimentData:
    """Run the operation sequentially

    Parameters
    ----------
    kwargs : dict
        Any keyword arguments that need to be supplied to the function

    Raises
    ------
    NoOpenJobsError
        Raised when there are no open jobs left
    """

    while True:
        job_number, experiment_sample = data.get_open_job()
        if job_number is None:
            logger.debug("No Open jobs left!")
            break

        try:
            if pass_id:
                experiment_sample: ExperimentSample = execute_fn(
                    experiment_sample=experiment_sample, id=job_number,
                    **kwargs)
            else:
                experiment_sample: ExperimentSample = execute_fn(
                    experiment_sample=experiment_sample, **kwargs)

            experiment_sample.store_experimentsample_references(
                idx=job_number)

            experiment_sample.mark('finished')

        except Exception:
            error_msg = f"Error in experiment_sample {job_number}: "
            error_traceback = traceback.format_exc()
            logger.error(f"{error_msg}\n{error_traceback}")
            experiment_sample.mark('error')
            continue

        finally:
            data.store_experimentsample(
                idx=job_number, experiment_sample=experiment_sample,
            )
    return data


def _evaluate_multiprocessing(
    execute_fn: Callable, data: ExperimentData, pass_id: bool,
        nodes: int = mp.cpu_count(), **kwargs) -> ExperimentData:
    options = []

    while True:
        job_number, experiment_sample = data.get_open_job()
        if job_number is None:
            break

        if pass_id:
            options.append(
                ({'experiment_sample': experiment_sample,
                    '_job_number': job_number, 'id': job_number, **kwargs},))
        else:
            options.append(
                ({'experiment_sample': experiment_sample,
                    '_job_number': job_number, **kwargs},))

    def f(options: dict[str, Any]) -> tuple[int, ExperimentSample, int]:
        job_number = options.pop('_job_number')
        try:

            logger.debug(
                f"Running experiment_sample "
                f"{job_number}")

            experiment_sample: ExperimentSample = execute_fn(**options)
            experiment_sample.store_experimentsample_references(
                idx=job_number)
            # exit_code = 0
            experiment_sample.mark('finished')

        except Exception:
            error_msg = f"Error in experiment_sample {job_number}: "
            error_traceback = traceback.format_exc()
            logger.error(f"{error_msg}\n{error_traceback}")
            experiment_sample.mark('error')
            # exit_code = 1

        finally:
            return (job_number, experiment_sample)

    with mp.Pool() as pool:
        # maybe implement pool.starmap_async ?
        _experiment_samples: list[
            tuple[int, ExperimentSample, int]] = pool.starmap(f, options)

    for job_number, experiment_sample in _experiment_samples:
        data.store_experimentsample(
            experiment_sample=experiment_sample,
            idx=job_number)

    return data


def _evaluate_cluster(
    execute_fn: Callable, data: ExperimentData, pass_id: bool,
        wait_for_creation: bool = False,
        max_tries: int = MAX_TRIES, **kwargs
) -> None:

    # Creat lockfile
    lockfile = FileLock(
        (data.project_dir / EXPERIMENTDATA_SUBFOLDER / LOCK_FILENAME
         ).with_suffix('.lock'))

    cluster_get_open_job = partial(
        get_open_job, experiment_data_type=type(data),
        project_dir=data.project_dir,
        wait_for_creation=wait_for_creation, max_tries=max_tries,
        lockfile=lockfile)
    cluster_store_experiment_sample = partial(
        store_experiment_sample, experiment_data_type=type(data),
        project_dir=data.project_dir,
        wait_for_creation=wait_for_creation, max_tries=max_tries,
        lockfile=lockfile)

    while True:
        job_number, experiment_sample = cluster_get_open_job()
        if job_number is None:
            logger.debug("No Open jobs left!")
            break

        try:
            if pass_id:
                experiment_sample: ExperimentSample = execute_fn(
                    experiment_sample=experiment_sample, id=job_number,
                    **kwargs)
            else:
                experiment_sample: ExperimentSample = execute_fn(
                    experiment_sample=experiment_sample, **kwargs)

            experiment_sample.store_experimentsample_references(
                idx=job_number)

            experiment_sample.mark('finished')

        except Exception:
            error_msg = f"Error in experiment_sample {job_number}: "
            error_traceback = traceback.format_exc()
            logger.error(f"{error_msg}\n{error_traceback}")
            experiment_sample.mark('error')
            continue

        finally:
            cluster_store_experiment_sample(
                idx=job_number, experiment_sample=experiment_sample)

    # Remove lockfile
    (data.project_dir / EXPERIMENTDATA_SUBFOLDER / LOCK_FILENAME
     ).with_suffix('.lock').unlink(missing_ok=True)


def _evaluate_mpi(
    execute_fn: Callable, comm, data: ExperimentData, pass_id: bool,
    wait_for_creation: bool = False,
    max_tries: int = MAX_TRIES, **kwargs
) -> None:
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        mpi_lock_manager(comm=comm, size=size)
    else:
        mpi_worker(comm=comm, data=data, execute_fn=execute_fn,
                   pass_id=pass_id,
                   wait_for_creation=wait_for_creation,
                   max_tries=max_tries,
                   **kwargs)


# =============================================================================


def get_open_job(experiment_data_type: type[ExperimentData],
                 project_dir: Path, lockfile: FileLock,
                 wait_for_creation: bool, max_tries: int,
                 ) -> tuple[int, ExperimentSample]:

    with lockfile:
        data: ExperimentData = experiment_data_type.from_file(
            project_dir=project_dir, wait_for_creation=wait_for_creation,
            max_tries=max_tries)

        idx, es = data.get_open_job()

        data.store(project_dir)

    return idx, es


def store_experiment_sample(
    experiment_data_type: type[ExperimentData],
        project_dir: Path, lockfile: FileLock, wait_for_creation: bool,
        max_tries: int, idx: int, experiment_sample: ExperimentSample) -> None:

    with lockfile:
        data: ExperimentData = experiment_data_type.from_file(
            project_dir=project_dir, wait_for_creation=wait_for_creation,
            max_tries=max_tries)

        data.store_experimentsample(experiment_sample=experiment_sample,
                                    idx=idx)
        data.store(project_dir)


def mpi_worker(
    comm, data: ExperimentData,
        execute_fn: Callable,
        pass_id: bool,
        wait_for_creation: bool = False,
        max_tries: int = MAX_TRIES, **kwargs
) -> None:

    cluster_get_open_job = partial(
        mpi_get_open_job, experiment_data_type=type(data),
        project_dir=data.project_dir,
        wait_for_creation=wait_for_creation, max_tries=max_tries,
        comm=comm)
    cluster_store_experiment_sample = partial(
        mpi_store_experiment_sample, experiment_data_type=type(data),
        project_dir=data.project_dir,
        wait_for_creation=wait_for_creation, max_tries=max_tries,
        comm=comm)

    while True:
        job_number, experiment_sample = cluster_get_open_job()
        if job_number is None:
            logger.debug("No Open jobs left!")
            break

        try:
            if pass_id:
                experiment_sample: ExperimentSample = execute_fn(
                    experiment_sample=experiment_sample, id=job_number,
                    **kwargs)
            else:
                experiment_sample: ExperimentSample = execute_fn(
                    experiment_sample=experiment_sample, **kwargs)

            experiment_sample.store_experimentsample_references(
                idx=job_number)

            experiment_sample.mark('finished')

        except Exception:
            error_msg = f"Error in experiment_sample {job_number}: "
            error_traceback = traceback.format_exc()
            logger.error(f"{error_msg}\n{error_traceback}")
            experiment_sample.mark('error')
            continue

        finally:
            cluster_store_experiment_sample(
                idx=job_number, experiment_sample=experiment_sample)

    mpi_terminate_worker(comm)
