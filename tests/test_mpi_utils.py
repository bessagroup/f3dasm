"""Tests for MPI utility constants and fallback behavior."""

import pytest

from f3dasm._src import mpi_utils
from f3dasm._src.mpi_utils import (
    LOCK_GRANTED,
    LOCK_RELEASE,
    LOCK_REQUEST,
    MASTER_RANK,
    TERMINATE,
    mpi_terminate_worker,
)

pytestmark = pytest.mark.smoke


class TestMPIConstants:
    def test_constants_are_distinct(self):
        tags = {LOCK_REQUEST, LOCK_GRANTED, LOCK_RELEASE, TERMINATE}
        assert len(tags) == 4

    def test_master_rank_is_zero(self):
        assert MASTER_RANK == 0


class TestMPIAvailableFlag:
    def test_mpi_available_is_bool(self):
        assert isinstance(mpi_utils.MPI_AVAILABLE, bool)


class TestMPILockManager:
    def test_raises_without_mpi(self):
        if not mpi_utils.MPI_AVAILABLE:
            with pytest.raises(RuntimeError, match="mpi4py is not installed"):
                mpi_utils.mpi_lock_manager(comm=None, size=2)


class TestMPITerminateWorker:
    def test_sends_terminate_tag(self):
        sent = []

        class FakeComm:
            def send(self, data, dest, tag):
                sent.append({"dest": dest, "tag": tag})

        comm = FakeComm()
        mpi_terminate_worker(comm)
        assert len(sent) == 1
        assert sent[0]["dest"] == MASTER_RANK
        assert sent[0]["tag"] == TERMINATE
