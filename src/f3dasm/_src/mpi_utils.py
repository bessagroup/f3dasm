"""
This module defines tools for using MPI in a distributed fashion.
"""
#                                                                       Modules
# =============================================================================

# Standard
from pathlib import Path

# Third-party
from mpi4py import MPI
from mpi4py.MPI import Comm

# Local
from .logger import logger

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Under development'
# =============================================================================
#
# =============================================================================

LOCK_REQUEST = 1
LOCK_GRANTED = 2
LOCK_RELEASE = 3
TERMINATE = 4

# =============================================================================


def mpi_lock_manager(comm: Comm, size: int):
    """Process rank 0 acts as the centralized lock manager."""
    lock_held_by = None  # Track which rank holds the lock
    request_queue = []    # Queue of processes waiting for the lock
    termination_count = 0  # Count how many workers have sent TERMINATE

    while termination_count < size - 1:  # All workers should send TERMINATE
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
            termination_count += 1  # Count termination messages

    logger.info("Lock manager terminating.")


def mpi_get_open_job(comm: Comm, experiment_data_type,
                     project_dir: Path, wait_for_creation: bool,
                     max_tries: int):

    logger.debug(f"Process {comm.Get_rank()} requesting lock")
    comm.send(None, dest=0, tag=LOCK_REQUEST)

    comm.recv(source=0, tag=LOCK_GRANTED)  # Wait until lock is granted
    logger.debug(f"Process {comm.Get_rank()} acquired lock")

    try:
        data = experiment_data_type.from_file(
            project_dir=project_dir, wait_for_creation=wait_for_creation,
            max_tries=max_tries)

        idx, es = data.get_open_job()

        data.store(project_dir)

    finally:
        logger.debug(f"Process {comm.Get_rank()} releasing lock")
        comm.send(None, dest=0, tag=LOCK_RELEASE)

    return idx, es


def mpi_store_experiment_sample(
    comm: Comm, experiment_data_type,
        project_dir: Path, wait_for_creation: bool,
        max_tries: int, idx: int, experiment_sample) -> None:

    logger.debug(f"Process {comm.Get_rank()} requesting lock")
    comm.send(None, dest=0, tag=LOCK_REQUEST)

    comm.recv(source=0, tag=LOCK_GRANTED)  # Wait until lock is granted
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
        comm.send(None, dest=0, tag=LOCK_RELEASE)


def mpi_terminate_worker(comm: Comm) -> None:
    comm.send(None, dest=0, tag=TERMINATE)
