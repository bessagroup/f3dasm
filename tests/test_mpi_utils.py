"""Tests for MPI utility constants and fallback behavior."""

import threading
from unittest.mock import MagicMock, patch

import pytest

from f3dasm._src import mpi_utils
from f3dasm._src.mpi_utils import (
    LOCK_GRANTED,
    LOCK_RELEASE,
    LOCK_REQUEST,
    MASTER_RANK,
    TERMINATE,
    mpi_get_open_job,
    mpi_store_experiment_sample,
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

    def test_lock_grant_and_release(self, monkeypatch):
        """Test that the lock manager grants and releases locks correctly."""
        import types

        monkeypatch.setattr(mpi_utils, "MPI_AVAILABLE", True)

        # Create a fake MPI module with Status class
        fake_mpi = types.ModuleType("fake_mpi")
        fake_mpi.ANY_SOURCE = -1
        fake_mpi.ANY_TAG = -1

        messages = [
            (1, LOCK_REQUEST),
            (1, LOCK_RELEASE),
            (1, TERMINATE),
        ]
        msg_iter = iter(messages)
        sent_messages = []

        class FakeStatus:
            def __init__(self):
                self._source = 0
                self._tag = 0

            def Get_source(self):
                return self._source

            def Get_tag(self):
                return self._tag

        fake_mpi.Status = FakeStatus

        def fake_recv(source, tag, status):
            src, t = next(msg_iter)
            status._source = src
            status._tag = t
            return None

        def fake_send(data, dest, tag):
            sent_messages.append({"dest": dest, "tag": tag})

        comm = MagicMock()
        comm.recv = fake_recv
        comm.send = fake_send

        monkeypatch.setattr(mpi_utils, "MPI", fake_mpi, raising=False)

        mpi_utils.mpi_lock_manager(comm=comm, size=2)

        assert any(
            m["dest"] == 1 and m["tag"] == LOCK_GRANTED for m in sent_messages
        )

    def test_lock_queuing(self, monkeypatch):
        """Test that a second worker queues when the lock is held."""
        import types

        monkeypatch.setattr(mpi_utils, "MPI_AVAILABLE", True)

        fake_mpi = types.ModuleType("fake_mpi")
        fake_mpi.ANY_SOURCE = -1
        fake_mpi.ANY_TAG = -1

        messages = [
            (1, LOCK_REQUEST),
            (2, LOCK_REQUEST),
            (1, LOCK_RELEASE),
            (2, LOCK_RELEASE),
            (1, TERMINATE),
            (2, TERMINATE),
        ]
        msg_iter = iter(messages)
        sent_messages = []

        class FakeStatus:
            def __init__(self):
                self._source = 0
                self._tag = 0

            def Get_source(self):
                return self._source

            def Get_tag(self):
                return self._tag

        fake_mpi.Status = FakeStatus

        def fake_recv(source, tag, status):
            src, t = next(msg_iter)
            status._source = src
            status._tag = t
            return None

        def fake_send(data, dest, tag):
            sent_messages.append({"dest": dest, "tag": tag})

        comm = MagicMock()
        comm.recv = fake_recv
        comm.send = fake_send

        monkeypatch.setattr(mpi_utils, "MPI", fake_mpi, raising=False)

        mpi_utils.mpi_lock_manager(comm=comm, size=3)

        grant_dests = [
            m["dest"] for m in sent_messages if m["tag"] == LOCK_GRANTED
        ]
        assert grant_dests == [1, 2]


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


class TestMPIGetOpenJob:
    def test_get_open_job_success(self, tmp_path):
        """Test mpi_get_open_job acquires lock, loads data, and releases."""
        from f3dasm import ExperimentData, create_sampler
        from f3dasm.design import Domain

        domain = Domain()
        domain.add_float("x", 0.0, 1.0)
        domain.add_output("y")
        data = ExperimentData(domain=domain)
        sampler = create_sampler("random", seed=42)
        data = sampler.call(data=data, n_samples=3)
        data.store(project_dir=tmp_path)

        sent = []

        class FakeComm:
            def Get_rank(self):
                return 1

            def send(self, data, dest, tag):
                sent.append({"dest": dest, "tag": tag})

            def recv(self, source, tag):
                return None

        comm = FakeComm()
        idx, es, domain = mpi_get_open_job(
            comm=comm,
            experiment_data_type=ExperimentData,
            project_dir=tmp_path,
            wait_for_creation=False,
            max_tries=5,
        )

        assert idx is not None
        # Should have sent LOCK_REQUEST and LOCK_RELEASE
        assert sent[0]["tag"] == LOCK_REQUEST
        assert sent[-1]["tag"] == LOCK_RELEASE

    def test_get_open_job_releases_on_exception(self, tmp_path):
        """Test lock is released even when an exception occurs."""
        sent = []

        class FakeComm:
            def Get_rank(self):
                return 1

            def send(self, data, dest, tag):
                sent.append({"dest": dest, "tag": tag})

            def recv(self, source, tag):
                return None

        fake_type = MagicMock()
        fake_type.from_file.side_effect = FileNotFoundError("test")

        comm = FakeComm()
        with pytest.raises(FileNotFoundError):
            mpi_get_open_job(
                comm=comm,
                experiment_data_type=fake_type,
                project_dir=tmp_path,
                wait_for_creation=False,
                max_tries=1,
            )

        # Lock release should still happen
        assert sent[-1]["tag"] == LOCK_RELEASE


class TestMPIStoreExperimentSample:
    def test_store_experiment_sample(self, tmp_path):
        """Test mpi_store_experiment_sample acquires lock and stores."""
        from f3dasm import ExperimentData, ExperimentSample, create_sampler
        from f3dasm.design import Domain

        domain = Domain()
        domain.add_float("x", 0.0, 1.0)
        domain.add_output("y")
        data = ExperimentData(domain=domain)
        sampler = create_sampler("random", seed=42)
        data = sampler.call(data=data, n_samples=3)
        data.store(project_dir=tmp_path)

        sample = data.get_experiment_sample(0)
        sample.store("y", 42.0, to_disk=False)
        sample.mark("finished")

        sent = []

        class FakeComm:
            def Get_rank(self):
                return 1

            def send(self, data, dest, tag):
                sent.append({"dest": dest, "tag": tag})

            def recv(self, source, tag):
                return None

        comm = FakeComm()
        mpi_store_experiment_sample(
            comm=comm,
            experiment_data_type=ExperimentData,
            project_dir=tmp_path,
            wait_for_creation=False,
            max_tries=5,
            idx=0,
            experiment_sample=sample,
            domain=domain,
        )

        assert sent[0]["tag"] == LOCK_REQUEST
        assert sent[-1]["tag"] == LOCK_RELEASE

    def test_store_releases_on_exception(self, tmp_path):
        """Test lock is released even when store raises."""
        sent = []

        class FakeComm:
            def Get_rank(self):
                return 1

            def send(self, data, dest, tag):
                sent.append({"dest": dest, "tag": tag})

            def recv(self, source, tag):
                return None

        fake_type = MagicMock()
        fake_type.from_file.side_effect = FileNotFoundError("test")

        comm = FakeComm()
        with pytest.raises(FileNotFoundError):
            mpi_store_experiment_sample(
                comm=comm,
                experiment_data_type=fake_type,
                project_dir=tmp_path,
                wait_for_creation=False,
                max_tries=1,
                idx=0,
                experiment_sample=MagicMock(),
                domain=MagicMock(),
            )

        assert sent[-1]["tag"] == LOCK_RELEASE
