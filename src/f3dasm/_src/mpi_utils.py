"""
This module defines tools for using MPI in a distributed fashion.
"""
#                                                                       Modules
# =============================================================================

# Standard
from pathlib import Path
from typing import TYPE_CHECKING

# Third-party
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False

if TYPE_CHECKING:
    from mpi4py.MPI import Comm
else:
    Comm = object

# Local
from .logger import logger

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Under development'
# =============================================================================
#
#                                                                     Constants
# =============================================================================

LOCK_REQUEST = 1
LOCK_GRANTED = 2
LOCK_RELEASE = 3
TERMINATE = 4

MASTER_RANK = 0

#                                                              MPI Lock manager
# =============================================================================


def mpi_lock_manager(comm: Comm, size: int):
    """
    Centralized lock manager process (rank 0) to handle MPI-based locking.

    Parameters
    ----------
    comm : Comm
        MPI communicator object.
    size : int
        Total number of processes in the MPI communicator.
    """
    if not MPI_AVAILABLE:
        raise RuntimeError(
            "mpi4py is not installed. Install it to use MPI features.")

    # Track which rank holds the lock
    lock_held_by = None

    # Queue of processes waiting for the lock
    request_queue = []

    # Count how many workers have sent TERMINATE
    termination_count = 0

    # Loop until all workers have sent TERMINATE
    while termination_count < size - 1:
        status = MPI.Status()
        _ = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        sender = status.Get_source()
        tag = status.Get_tag()

        if tag == LOCK_REQUEST:
            if lock_held_by is None:
                lock_held_by = sender
                comm.send(None, dest=sender, tag=LOCK_GRANTED)
                logger.debug(f"Lock granted to process {sender}")
            else:
                request_queue.append(sender)
                logger.debug(f"Process {sender} added to queue")

        elif tag == LOCK_RELEASE:
            if lock_held_by == sender:
                lock_held_by = None
                if request_queue:
                    next_in_line = request_queue.pop(0)
                    lock_held_by = next_in_line
                    comm.send(None, dest=next_in_line, tag=LOCK_GRANTED)
                    logger.debug(f"Lock granted to process {next_in_line}")

        elif tag == TERMINATE:
            # Increment termination messages by one
            termination_count += 1
            logger.debug(f"Process {sender} sent termination signal")

    logger.info("Lock manager terminating.")

#                                                              MPI Worker tools
# =============================================================================


def mpi_get_open_job(comm: Comm, experiment_data_type,
                     project_dir: Path, wait_for_creation: bool,
                     max_tries: int):
    """
    Request and acquire an MPI lock to retrieve an open job
    from the experiment data.

    Parameters
    ----------
    comm : Comm
        MPI communicator object.
    experiment_data_type : type
        Class type for handling experiment data.
    project_dir : Path
        Path to the project directory where experiment data is stored.
    wait_for_creation : bool
        Whether to wait for the experiment data file to be created.
    max_tries : int
        Maximum number of attempts to access the experiment data.

    Returns
    -------
    tuple
        Index of the open job and the experiment sample data.
    """
    logger.debug(f"Process {comm.Get_rank()} requesting lock")
    comm.send(None, dest=0, tag=LOCK_REQUEST)

    # Wait until lock is granted
    comm.recv(source=MASTER_RANK, tag=LOCK_GRANTED)
    logger.debug(f"Process {comm.Get_rank()} acquired lock")

    try:
        data = experiment_data_type.from_file(
            project_dir=project_dir, wait_for_creation=wait_for_creation,
            max_tries=max_tries)

        idx, es = data.get_open_job()

        data.store(project_dir)

    finally:
        logger.debug(f"Process {comm.Get_rank()} releasing lock")
        comm.send(None, dest=MASTER_RANK, tag=LOCK_RELEASE)

    return idx, es


def mpi_store_experiment_sample(
    comm: Comm, experiment_data_type,
        project_dir: Path, wait_for_creation: bool,
        max_tries: int, idx: int, experiment_sample) -> None:
    """
    Request and acquire an MPI lock to store an experiment sample.

    Parameters
    ----------
    comm : Comm
        MPI communicator object.
    experiment_data_type : type
        Class type for handling experiment data.
    project_dir : Path
        Path to the project directory where experiment data is stored.
    wait_for_creation : bool
        Whether to wait for the experiment data file to be created.
    max_tries : int
        Maximum number of attempts to access the experiment data.
    idx : int
        Index of the experiment sample.
    experiment_sample : Any
        The experiment sample data to be stored.
    """
    logger.debug(f"Process {comm.Get_rank()} requesting lock")
    comm.send(None, dest=MASTER_RANK, tag=LOCK_REQUEST)

    # Wait until lock is granted
    comm.recv(source=MASTER_RANK, tag=LOCK_GRANTED)
    logger.debug(f"Process {comm.Get_rank()} acquired lock")

    try:
        data = experiment_data_type.from_file(
            project_dir=project_dir, wait_for_creation=wait_for_creation,
            max_tries=max_tries)

        data.store_experimentsample(experiment_sample=experiment_sample,
                                    idx=idx)
        data.store(project_dir)

    finally:
        logger.debug(f"Process {comm.Get_rank()} releasing lock")
        comm.send(None, dest=MASTER_RANK, tag=LOCK_RELEASE)


def mpi_terminate_worker(comm: Comm) -> None:
    """
    Send a termination signal to the MPI lock manager.

    Parameters
    ----------
    comm : Comm
        MPI communicator object.
    """
    comm.send(None, dest=MASTER_RANK, tag=TERMINATE)
