"""
This module contains the DataGenerator abstraction.
"""
#                                                                       Modules
# =============================================================================

from __future__ import annotations

import logging

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
from .design.domain import Domain
from .experimentdata import ExperimentData, _store
from .experimentsample import ExperimentSample
from .mpi_utils import (
    mpi_get_open_job,
    mpi_lock_manager,
    mpi_store_experiment_sample,
    mpi_terminate_worker,
)

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Alpha"
# =============================================================================
#
# =============================================================================

logger = logging.getLogger("f3dasm")

# =============================================================================


def _run_sample(
    execute_fn: Callable[..., ExperimentSample],
    experiment_sample: ExperimentSample,
    domain: Domain,
    job_number: int | None = None,
    pass_id: bool = False,
    **kwargs: Any,
) -> tuple[ExperimentSample, Domain]:
    """Run `execute_fn` on a single ExperimentSample, handle exceptions and
    mark/store status on the sample object before returning it.

    This helper is purposely simple so it is picklable for multiprocessing.

    Parameters
    ----------
    execute_fn : Callable[..., ExperimentSample]
        The function to execute on the experiment sample.
    experiment_sample : ExperimentSample
        The experiment sample to process.
    domain : Domain
        The domain of the experiment data.
    job_number : int or None, optional
        The index of the job, passed to `execute_fn` when `pass_id` is True,
        by default None.
    pass_id : bool, optional
        Whether to pass `job_number` as the `id` keyword argument to
        `execute_fn`, by default False.
    **kwargs : Any
        Additional keyword arguments forwarded to `execute_fn`.

    Returns
    -------
    tuple[ExperimentSample, Domain]
        The updated experiment sample (marked 'finished' or 'error') and the
        (possibly updated) domain.
    """
    try:
        logger.debug(f"Running experiment_sample {job_number}")
        if pass_id and job_number is not None:
            experiment_sample = execute_fn(
                experiment_sample=experiment_sample, id=job_number, **kwargs
            )
        else:
            experiment_sample = execute_fn(
                experiment_sample=experiment_sample, **kwargs
            )

        experiment_sample, domain = _store(
            experiment_sample=experiment_sample, idx=job_number, domain=domain
        )
        experiment_sample.mark("finished")

    except Exception:
        error_msg = f"Error in experiment_sample {job_number}: "
        error_traceback = traceback.format_exc()
        logger.error(f"{error_msg}\n{error_traceback}")
        experiment_sample.mark("error")

    return experiment_sample, domain


# =========================================================================


def evaluate_sequential(
    execute_fn: Callable[..., ExperimentSample],
    data: ExperimentData,
    pass_id: bool,
    **kwargs,
) -> ExperimentData:
    """Run the operation sequentially

    Parameters
    ----------
    execute_fn : Callable[..., ExperimentSample]
        The function to be executed on each ExperimentSample
    data : ExperimentData
        The ExperimentData object containing the samples to be processed
    pass_id : bool
        Whether to pass the job index to the execute function
    kwargs : dict
        Any keyword arguments that need to be supplied to the function

    Returns
    -------
    ExperimentData
        The updated ExperimentData object
    """

    while True:
        job_number, experiment_sample, domain = data.get_open_job()
        if job_number is None:
            logger.debug("No open jobs left!")
            break

        experiment_sample, domain = _run_sample(
            execute_fn=execute_fn,
            experiment_sample=experiment_sample,
            domain=domain,
            job_number=job_number,
            pass_id=pass_id,
            **kwargs,
        )

        data.domain = domain
        data.data[job_number] = experiment_sample

    return data


def evaluate_multiprocessing(
    execute_fn: Callable[..., ExperimentSample],
    data: ExperimentData,
    pass_id: bool,
    nodes: int = mp.cpu_count(),
    **kwargs,
) -> ExperimentData:
    """Run the operation using multiprocessing

    Parameters
    ----------
    execute_fn : Callable[..., ExperimentSample]
        The function to be executed on each ExperimentSample
    data : ExperimentData
        The ExperimentData object containing the samples to be processed
    pass_id : bool
        Whether to pass the job index to the execute function
    nodes : int, optional
        The number of parallel processes to use, by default mp.cpu_count()
    kwargs : dict
        Any keyword arguments that need to be supplied to the function

    Returns
    -------
    ExperimentData
        The updated ExperimentData object
    """
    work_items: list[dict[str, Any]] = []

    while True:
        job_number, experiment_sample, domain = data.get_open_job()
        if job_number is None:
            logger.debug("No open jobs left!")
            break

        item = {
            "experiment_sample": experiment_sample,
            "job_number": job_number,
            "domain": domain,
            **kwargs,
        }

        work_items.append(item)

    def _worker(
        options: dict[str, Any],
    ) -> tuple[int, ExperimentSample, Domain]:
        es, domain = _run_sample(
            execute_fn=execute_fn, pass_id=pass_id, **options
        )
        return options["job_number"], es, domain

    if work_items:
        with mp.Pool(nodes) as pool:
            # maybe implement pool.starmap_async ?
            results: list[tuple[int, ExperimentSample, Domain]] = pool.map(
                _worker, work_items
            )

        for job_number, experiment_sample, domain in results:
            # data.store_experimentsample(
            #     experiment_sample=experiment_sample,
            #     idx=job_number, domain=domain)

            data.domain = domain
            data.data[job_number] = experiment_sample

    return data


def evaluate_cluster(
    execute_fn: Callable[..., ExperimentSample],
    data: ExperimentData,
    pass_id: bool,
    wait_for_creation: bool = False,
    max_tries: int = MAX_TRIES,
    **kwargs,
) -> None:
    """
    Run the operation on a cluster using file locks to manage access to the
    ExperimentData.

    Parameters
    ----------

    execute_fn : Callable[..., ExperimentSample]
        The function to be executed on each ExperimentSample
    data : ExperimentData
        The ExperimentData object containing the samples to be processed
    pass_id : bool
        Whether to pass the job index to the execute function
    wait_for_creation : bool, optional
        Whether to wait for the ExperimentData file to be created, by default
        False
    max_tries : int, optional
        The maximum number of tries to access the ExperimentData file, by
        default MAX_TRIES
    kwargs : dict
        Any keyword arguments that need to be supplied to the function
    """

    # Creat lockfile
    lock_path = (
        data._project_dir / EXPERIMENTDATA_SUBFOLDER / LOCK_FILENAME
    ).with_suffix(".lock")
    lockfile = FileLock(lock_path)

    cluster_get_open_job = partial(
        _get_open_job,
        project_dir=data._project_dir,
        wait_for_creation=wait_for_creation,
        max_tries=max_tries,
        lockfile=lockfile,
    )
    cluster_store_experiment_sample = partial(
        _store_experiment_sample,
        project_dir=data._project_dir,
        wait_for_creation=wait_for_creation,
        max_tries=max_tries,
        lockfile=lockfile,
    )

    while True:
        job_number, experiment_sample, domain = cluster_get_open_job()
        if job_number is None:
            logger.debug("No open jobs left!")
            break

        experiment_sample, domain = _run_sample(
            execute_fn=execute_fn,
            experiment_sample=experiment_sample,
            domain=domain,
            job_number=job_number,
            pass_id=pass_id,
            **kwargs,
        )

        cluster_store_experiment_sample(
            idx=job_number, experiment_sample=experiment_sample, domain=domain
        )

    # Remove lockfile
    lock_path.unlink(missing_ok=True)


def evaluate_mpi(
    execute_fn: Callable[..., ExperimentSample],
    data: ExperimentData,
    pass_id: bool,
    wait_for_creation: bool = False,
    max_tries: int = MAX_TRIES,
    **kwargs,
) -> None:
    """
    Run the operation on a cluster using MPI to manage access to the
    ExperimentData.

    Parameters
    ----------
    execute_fn : Callable[..., ExperimentSample]
        The function to be executed on each ExperimentSample
    data : ExperimentData
        The ExperimentData object containing the samples to be processed
    pass_id : bool
        Whether to pass the job index to the execute function
    wait_for_creation : bool, optional
        Whether to wait for the ExperimentData file to be created, by default
        False
    max_tries : int, optional
        The maximum number of tries to access the ExperimentData file, by
        default MAX_TRIES
    kwargs : dict
        Any keyword arguments that need to be supplied to the function.
        Must include ``comm`` (the ``MPI.Comm`` instance) -- pulled out
        here rather than being a positional parameter so every
        ``evaluate_*`` function shares the same
        ``(execute_fn, data, pass_id, **kwargs)`` shape (issue #309).

    Raises
    ------
    TypeError
        If ``comm`` was not supplied in ``kwargs``.
    """
    try:
        comm = kwargs.pop("comm")
    except KeyError as exc:
        raise TypeError(
            "evaluate_mpi requires the MPI communicator via "
            "`comm=...`; pass it as a keyword argument."
        ) from exc

    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        mpi_lock_manager(comm=comm, size=size)
    else:
        mpi_worker(
            comm=comm,
            data=data,
            execute_fn=execute_fn,
            pass_id=pass_id,
            wait_for_creation=wait_for_creation,
            max_tries=max_tries,
            **kwargs,
        )


# =============================================================================


def evaluate_cluster_array(
    execute_fn: Callable[..., ExperimentSample],
    data: ExperimentData,
    pass_id: bool,
    **kwargs,
) -> None:
    """
    Run the operation on a cluster using file locks to manage access to the
    ExperimentData.

    Parameters
    ----------
    execute_fn : Callable[..., ExperimentSample]
        The function to be executed on each ExperimentSample.
    data : ExperimentData
        The ExperimentData object containing the samples to be processed.
    pass_id : bool
        Whether to pass the job index to the execute function.
    **kwargs : dict
        Any keyword arguments that need to be supplied to the function.
        Must include ``job_number`` (the array index to run) -- pulled
        out of ``kwargs`` here rather than being a positional parameter
        so every ``evaluate_*`` function shares the same
        ``(execute_fn, data, pass_id, **kwargs)`` shape (issue #309).

    Raises
    ------
    TypeError
        If ``job_number`` was not supplied in ``kwargs``.
    """
    try:
        job_number = kwargs.pop("job_number")
    except KeyError as exc:
        raise TypeError(
            "evaluate_cluster_array requires the array index via "
            "`job_number=...`; pass it as a keyword argument."
        ) from exc

    # Retrieve the experiment sample
    experiment_sample = data.get_experiment_sample(job_number)

    # Mark as in progress
    experiment_sample.mark("in_progress")

    # Store the experiment sample to disk
    experiment_sample.store_as_json(idx=job_number)

    # Run the experiment sample
    experiment_sample, _ = _run_sample(
        execute_fn=execute_fn,
        experiment_sample=experiment_sample,
        domain=data.domain,
        job_number=job_number,
        pass_id=pass_id,
        **kwargs,
    )

    # Update the experiment sample to disk
    experiment_sample.store_as_json(idx=job_number)


# =============================================================================


def _get_open_job(
    project_dir: Path,
    lockfile: FileLock,
    wait_for_creation: bool,
    max_tries: int,
) -> tuple[int, ExperimentSample]:
    """Load ExperimentData under a file lock and retrieve an open job.

    Parameters
    ----------
    project_dir : Path
        Path to the project directory where experiment data is stored.
    lockfile : FileLock
        File lock used to serialise access to the experiment data on disk.
    wait_for_creation : bool
        Whether to wait for the experiment data file to be created if it does
        not yet exist.
    max_tries : int
        Maximum number of attempts to access the experiment data file.

    Returns
    -------
    tuple[int, ExperimentSample, Domain]
        The job index, the experiment sample, and the domain.
    """
    with lockfile:
        data = ExperimentData.from_file(
            project_dir=project_dir,
            wait_for_creation=wait_for_creation,
            max_tries=max_tries,
        )

        idx, es, domain = data.get_open_job()

        data.store(project_dir)

    return idx, es, domain


def _store_experiment_sample(
    project_dir: Path,
    lockfile: FileLock,
    wait_for_creation: bool,
    max_tries: int,
    idx: int,
    experiment_sample: ExperimentSample,
    domain: Domain,
) -> None:
    """Store a processed experiment sample to disk under a file lock.

    Parameters
    ----------
    project_dir : Path
        Path to the project directory where experiment data is stored.
    lockfile : FileLock
        File lock used to serialise access to the experiment data on disk.
    wait_for_creation : bool
        Whether to wait for the experiment data file to be created if it does
        not yet exist.
    max_tries : int
        Maximum number of attempts to access the experiment data file.
    idx : int
        Index of the experiment sample within the ExperimentData.
    experiment_sample : ExperimentSample
        The processed experiment sample to write back to disk.
    domain : Domain
        The (possibly updated) domain to persist alongside the data.
    """
    with lockfile:
        data = ExperimentData.from_file(
            project_dir=project_dir,
            wait_for_creation=wait_for_creation,
            max_tries=max_tries,
        )

        data.domain = domain
        data.data[idx] = experiment_sample
        data.store(project_dir)


def _get_domain(
    project_dir: Path,
    lockfile: FileLock,
    wait_for_creation: bool,
    max_tries: int,
) -> Domain:
    """Load ExperimentData under a file lock and return its domain.

    Parameters
    ----------
    project_dir : Path
        Path to the project directory where experiment data is stored.
    lockfile : FileLock
        File lock used to serialise access to the experiment data on disk.
    wait_for_creation : bool
        Whether to wait for the experiment data file to be created if it does
        not yet exist.
    max_tries : int
        Maximum number of attempts to access the experiment data file.

    Returns
    -------
    Domain
        The domain of the loaded experiment data.
    """
    with lockfile:
        data = ExperimentData.from_file(
            project_dir=project_dir,
            wait_for_creation=wait_for_creation,
            max_tries=max_tries,
        )

        domain = data.domain

    return domain


def mpi_worker(
    comm,
    data: ExperimentData,
    execute_fn: Callable,
    pass_id: bool,
    wait_for_creation: bool = False,
    max_tries: int = MAX_TRIES,
    **kwargs,
) -> None:
    """Run the execution loop for an MPI worker process.

    Repeatedly fetches an open job via MPI-coordinated locking, runs
    `execute_fn` on it, and writes the result back until no open jobs remain.

    Parameters
    ----------
    comm : MPI.Comm
        The MPI communicator.
    data : ExperimentData
        The experiment data object (used for project directory and type info).
    execute_fn : Callable
        The function to execute on each ExperimentSample.
    pass_id : bool
        Whether to pass the job index as `id` to `execute_fn`.
    wait_for_creation : bool, optional
        Whether to wait for the experiment data file to be created if it does
        not yet exist, by default False.
    max_tries : int, optional
        Maximum number of attempts to access the experiment data file, by
        default MAX_TRIES.
    **kwargs : dict
        Additional keyword arguments forwarded to `execute_fn`.
    """
    cluster_get_open_job = partial(
        mpi_get_open_job,
        experiment_data_type=ExperimentData,
        project_dir=data._project_dir,
        wait_for_creation=wait_for_creation,
        max_tries=max_tries,
        comm=comm,
    )
    cluster_store_experiment_sample = partial(
        mpi_store_experiment_sample,
        experiment_data_type=ExperimentData,
        project_dir=data._project_dir,
        wait_for_creation=wait_for_creation,
        max_tries=max_tries,
        comm=comm,
    )

    while True:
        job_number, experiment_sample, domain = cluster_get_open_job()
        if job_number is None:
            logger.debug("No open jobs left!")
            break

        experiment_sample, domain = _run_sample(
            execute_fn=execute_fn,
            experiment_sample=experiment_sample,
            domain=domain,
            job_number=job_number,
            pass_id=pass_id,
            **kwargs,
        )

        cluster_store_experiment_sample(
            idx=job_number, experiment_sample=experiment_sample, domain=domain
        )

    mpi_terminate_worker(comm)
