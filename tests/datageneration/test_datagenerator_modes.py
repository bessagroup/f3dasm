"""Tests for DataGenerator execution modes and _run_sample."""

from unittest.mock import MagicMock, patch

import pytest

from f3dasm import ExperimentData, ExperimentSample, create_sampler
from f3dasm._src.datagenerator import (
    _run_sample,
    evaluate_cluster,
    evaluate_cluster_array,
    evaluate_mpi,
    evaluate_multiprocessing,
    evaluate_sequential,
    mpi_worker,
)
from f3dasm._src.experimentsample import JobStatus
from f3dasm.design import Domain

pytestmark = pytest.mark.smoke


@pytest.fixture
def simple_domain():
    d = Domain()
    d.add_float(name="x", low=0.0, high=1.0)
    d.add_output(name="y")
    return d


@pytest.fixture
def simple_data(simple_domain):
    data = ExperimentData(domain=simple_domain)
    sampler = create_sampler("random", seed=42)
    data = sampler.call(data=data, n_samples=3)
    return data


@pytest.fixture
def stored_data(simple_data, tmp_path):
    """Simple data stored on disk for cluster/file-lock tests."""
    simple_data.set_project_dir(tmp_path, in_place=True)
    simple_data.store(project_dir=tmp_path)
    return simple_data


def _square_fn(experiment_sample: ExperimentSample, **kwargs):
    x = experiment_sample.get("x")
    experiment_sample.store("y", x**2, to_disk=False)
    return experiment_sample


def _failing_fn(experiment_sample: ExperimentSample, **kwargs):
    raise RuntimeError("intentional error")


def _fn_with_id(experiment_sample: ExperimentSample, id=None, **kwargs):
    experiment_sample.store("y", float(id), to_disk=False)
    return experiment_sample


class TestRunSample:
    def test_successful_execution(self, simple_domain):
        es = ExperimentSample(
            _input_data={"x": 0.5}, _output_data={}, job_status="OPEN"
        )
        result_es, result_domain = _run_sample(
            execute_fn=_square_fn,
            experiment_sample=es,
            domain=simple_domain,
            job_number=0,
        )
        assert result_es.job_status == JobStatus.FINISHED

    def test_error_marks_error_status(self, simple_domain):
        es = ExperimentSample(
            _input_data={"x": 0.5}, _output_data={}, job_status="OPEN"
        )
        result_es, _ = _run_sample(
            execute_fn=_failing_fn,
            experiment_sample=es,
            domain=simple_domain,
            job_number=0,
        )
        assert result_es.job_status == JobStatus.ERROR

    def test_pass_id_true(self, simple_domain):
        es = ExperimentSample(
            _input_data={"x": 0.5}, _output_data={}, job_status="OPEN"
        )
        result_es, _ = _run_sample(
            execute_fn=_fn_with_id,
            experiment_sample=es,
            domain=simple_domain,
            job_number=7,
            pass_id=True,
        )
        assert result_es.job_status == JobStatus.FINISHED

    def test_pass_id_false(self, simple_domain):
        es = ExperimentSample(
            _input_data={"x": 0.5}, _output_data={}, job_status="OPEN"
        )
        result_es, _ = _run_sample(
            execute_fn=_square_fn,
            experiment_sample=es,
            domain=simple_domain,
            job_number=0,
            pass_id=False,
        )
        assert result_es.job_status == JobStatus.FINISHED


class TestEvaluateSequential:
    def test_processes_all_samples(self, simple_data):
        result = evaluate_sequential(
            execute_fn=_square_fn,
            data=simple_data,
            pass_id=False,
        )
        for _, sample in result.data.items():
            assert sample.job_status == JobStatus.FINISHED

    def test_error_does_not_stop_processing(self, simple_data):
        result = evaluate_sequential(
            execute_fn=_failing_fn,
            data=simple_data,
            pass_id=False,
        )
        for _, sample in result.data.items():
            assert sample.job_status == JobStatus.ERROR


class TestEvaluateMultiprocessing:
    def test_multiprocessing_processes_all(self, simple_data):
        result = evaluate_multiprocessing(
            execute_fn=_square_fn,
            data=simple_data,
            pass_id=False,
            nodes=1,
        )
        for _, sample in result.data.items():
            assert sample.job_status == JobStatus.FINISHED

    def test_multiprocessing_empty_data(self, simple_domain):
        data = ExperimentData(domain=simple_domain)
        result = evaluate_multiprocessing(
            execute_fn=_square_fn,
            data=data,
            pass_id=False,
            nodes=1,
        )
        assert len(result) == 0

    def test_multiprocessing_with_error_fn(self, simple_data):
        result = evaluate_multiprocessing(
            execute_fn=_failing_fn,
            data=simple_data,
            pass_id=False,
            nodes=1,
        )
        for _, sample in result.data.items():
            assert sample.job_status == JobStatus.ERROR


class TestEvaluateCluster:
    def test_cluster_processes_all(self, stored_data, tmp_path):
        evaluate_cluster(
            execute_fn=_square_fn,
            data=stored_data,
            pass_id=False,
        )
        # Reload from disk to verify
        loaded = ExperimentData.from_file(project_dir=tmp_path)
        for _, sample in loaded:
            assert sample.job_status == JobStatus.FINISHED

    def test_cluster_with_error_fn(self, stored_data, tmp_path):
        evaluate_cluster(
            execute_fn=_failing_fn,
            data=stored_data,
            pass_id=False,
        )
        loaded = ExperimentData.from_file(project_dir=tmp_path)
        for _, sample in loaded:
            assert sample.job_status == JobStatus.ERROR


class TestEvaluateClusterArray:
    def test_cluster_array_single_job(self, stored_data, tmp_path):
        from f3dasm._src._io import EXPERIMENTSAMPLE_SUBFOLDER

        # Ensure sample project_dir points to tmp_path
        for es in stored_data.data.values():
            es.project_dir = tmp_path

        evaluate_cluster_array(
            execute_fn=_square_fn,
            data=stored_data,
            job_number=0,
            pass_id=False,
        )
        # Should have created a JSON file
        subfolder = tmp_path / EXPERIMENTSAMPLE_SUBFOLDER
        json_files = list(subfolder.glob("*.json"))
        assert len(json_files) >= 1

    def test_cluster_array_with_error(self, stored_data, tmp_path):
        from f3dasm._src._io import EXPERIMENTSAMPLE_SUBFOLDER

        for es in stored_data.data.values():
            es.project_dir = tmp_path

        evaluate_cluster_array(
            execute_fn=_failing_fn,
            data=stored_data,
            job_number=0,
            pass_id=False,
        )
        # JSON should still be written even on error
        subfolder = tmp_path / EXPERIMENTSAMPLE_SUBFOLDER
        json_files = list(subfolder.glob("*.json"))
        assert len(json_files) >= 1


class TestEvaluateMPI:
    def test_mpi_rank_zero_calls_lock_manager(self, simple_data):
        comm = MagicMock()
        comm.Get_rank.return_value = 0
        comm.Get_size.return_value = 2

        with patch("f3dasm._src.datagenerator.mpi_lock_manager") as mock_lock:
            evaluate_mpi(
                execute_fn=_square_fn,
                comm=comm,
                data=simple_data,
                pass_id=False,
            )
            mock_lock.assert_called_once_with(comm=comm, size=2)

    def test_mpi_nonzero_rank_calls_worker(self, simple_data):
        comm = MagicMock()
        comm.Get_rank.return_value = 1
        comm.Get_size.return_value = 2

        with patch("f3dasm._src.datagenerator.mpi_worker") as mock_worker:
            evaluate_mpi(
                execute_fn=_square_fn,
                comm=comm,
                data=simple_data,
                pass_id=False,
            )
            mock_worker.assert_called_once()


class TestMPIWorker:
    def test_mpi_worker_processes_jobs_until_none(self, simple_data):
        comm = MagicMock()
        sample = simple_data.get_experiment_sample(0)

        call_count = 0

        def fake_get_open_job(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return 0, sample, simple_data.domain
            return None, ExperimentSample(), simple_data.domain

        with (
            patch(
                "f3dasm._src.datagenerator.mpi_get_open_job",
                side_effect=fake_get_open_job,
            ),
            patch(
                "f3dasm._src.datagenerator.mpi_store_experiment_sample"
            ) as mock_store,
            patch(
                "f3dasm._src.datagenerator.mpi_terminate_worker"
            ) as mock_terminate,
        ):
            mpi_worker(
                comm=comm,
                data=simple_data,
                execute_fn=_square_fn,
                pass_id=False,
            )
            mock_store.assert_called_once()
            mock_terminate.assert_called_once_with(comm)
