"""Extended tests for DataGenerator: _run_sample, error handling, etc."""

import numpy as np
import pytest

from f3dasm import ExperimentData, create_datagenerator, datagenerator
from f3dasm._src.datagenerator import _run_sample, evaluate_sequential
from f3dasm._src.experimentsample import ExperimentSample, JobStatus
from f3dasm.design import Domain

pytestmark = pytest.mark.smoke


# ======================= _run_sample =======================


def test_run_sample_success():
    """Successful execution marks sample as finished."""

    def execute_fn(experiment_sample, **kwargs):
        experiment_sample.store("y", 42.0)
        return experiment_sample

    domain = Domain()
    domain.add_parameter("x0")
    sample = ExperimentSample(_input_data={"x0": 1.0})
    result_sample, result_domain = _run_sample(
        execute_fn=execute_fn,
        experiment_sample=sample,
        domain=domain,
        job_number=0,
    )
    assert result_sample.job_status == JobStatus.FINISHED
    assert result_sample.output_data["y"] == 42.0


def test_run_sample_error_marks_status():
    """An exception in execute_fn marks sample as error."""

    def failing_fn(experiment_sample, **kwargs):
        raise RuntimeError("Intentional failure")

    domain = Domain()
    domain.add_parameter("x0")
    sample = ExperimentSample(_input_data={"x0": 1.0})
    result_sample, result_domain = _run_sample(
        execute_fn=failing_fn,
        experiment_sample=sample,
        domain=domain,
        job_number=0,
    )
    assert result_sample.job_status == JobStatus.ERROR


def test_run_sample_with_pass_id():
    """pass_id=True passes the job number to execute_fn."""

    def execute_with_id(experiment_sample, id=None, **kwargs):
        experiment_sample.store("job_id", float(id))
        return experiment_sample

    domain = Domain()
    domain.add_parameter("x0")
    sample = ExperimentSample(_input_data={"x0": 1.0})
    result_sample, _ = _run_sample(
        execute_fn=execute_with_id,
        experiment_sample=sample,
        domain=domain,
        job_number=7,
        pass_id=True,
    )
    assert result_sample.job_status == JobStatus.FINISHED
    assert result_sample.output_data["job_id"] == 7.0


# ======================= evaluate_sequential =======================


def test_evaluate_sequential_processes_all_samples():
    """evaluate_sequential should process all open samples."""

    def execute_fn(experiment_sample, **kwargs):
        x0 = experiment_sample.input_data["x0"]
        experiment_sample.store("y", x0 * 2)
        return experiment_sample

    data = ExperimentData(input_data=[{"x0": 1.0}, {"x0": 2.0}, {"x0": 3.0}])
    result = evaluate_sequential(
        execute_fn=execute_fn, data=data, pass_id=False
    )
    assert result.is_all_finished()
    _, df_out = result.to_pandas()
    np.testing.assert_array_equal(df_out["y"].values, [2.0, 4.0, 6.0])


# ======================= create_datagenerator =======================


def test_create_datagenerator_from_string():
    """Create a DataGenerator from a benchmark function name."""
    gen = create_datagenerator(data_generator="sphere", output_names="y")
    sample = ExperimentSample(_input_data={"x": np.array([1.0, 1.0])})
    result = gen.execute(sample)
    assert result.output_data["y"] == 2.0


def test_datagenerator_decorator_with_kwargs():
    """Extra kwargs are passed through to the decorated function."""

    @datagenerator(output_names="y")
    def f(x0, scale=1.0):
        return x0 * scale

    data = ExperimentData(input_data=[{"x0": 5.0}])
    result = f.call(data, scale=10.0)
    _, df_out = result.to_pandas()
    assert df_out["y"].iloc[0] == 50.0


def test_datagenerator_decorator_default_params():
    """Default parameters in the function are used."""

    @datagenerator(output_names="y")
    def f(x0, bias=100.0):
        return x0 + bias

    data = ExperimentData(input_data=[{"x0": 5.0}])
    result = f.call(data)
    _, df_out = result.to_pandas()
    assert df_out["y"].iloc[0] == 105.0
