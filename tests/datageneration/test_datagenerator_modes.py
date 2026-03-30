"""Tests for DataGenerator execution modes and _run_sample."""

import pytest

from f3dasm import ExperimentData, ExperimentSample, create_sampler
from f3dasm._src.datagenerator import _run_sample, evaluate_sequential
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
