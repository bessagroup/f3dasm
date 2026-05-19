"""Regression tests for issue #247: parallel workers must see different
random numbers from `np.random.*` and `random.*`."""

import random

import numpy as np
import pytest

from f3dasm import ExperimentData, ExperimentSample, create_sampler
from f3dasm._src.datagenerator import (
    _reseed_for_job,
    _run_sample,
    evaluate_multiprocessing,
    evaluate_sequential,
)
from f3dasm.design import Domain

pytestmark = pytest.mark.smoke


def _draw_globals_fn(experiment_sample: ExperimentSample, **kwargs):
    """Record one draw from each legacy global RNG so a test can check
    that parallel workers do not see identical sequences."""
    experiment_sample.store(
        "np_draw", float(np.random.random()), to_disk=False
    )
    experiment_sample.store("py_draw", random.random(), to_disk=False)
    return experiment_sample


@pytest.fixture
def domain_with_outputs():
    d = Domain()
    d.add_float(name="x", low=0.0, high=1.0)
    d.add_output(name="np_draw")
    d.add_output(name="py_draw")
    return d


@pytest.fixture
def four_sample_data(domain_with_outputs):
    data = ExperimentData(domain=domain_with_outputs)
    return create_sampler("random", seed=42).call(data=data, n_samples=4)


class TestReseedForJob:
    def test_distinct_seeds_yield_distinct_draws(self):
        seen = set()
        for job_number in range(8):
            _reseed_for_job(job_number)
            seen.add((float(np.random.random()), random.random()))
        # All eight (np, py) pairs must differ — that's the property the
        # parallel workers rely on.
        assert len(seen) == 8

    def test_same_seed_is_reproducible(self):
        _reseed_for_job(5)
        a_np, a_py = float(np.random.random()), random.random()
        _reseed_for_job(5)
        b_np, b_py = float(np.random.random()), random.random()
        assert a_np == b_np
        assert a_py == b_py

    def test_none_job_number_is_a_noop(self):
        # When no stable entropy source is known we should leave the
        # global RNG state untouched. Lock current state, call helper,
        # confirm the next draw matches what we'd have gotten anyway.
        np.random.seed(123)
        random.seed(123)
        expected_np = float(np.random.random())
        expected_py = random.random()

        np.random.seed(123)
        random.seed(123)
        _reseed_for_job(None)
        assert float(np.random.random()) == expected_np
        assert random.random() == expected_py


def test_evaluate_sequential_workers_see_distinct_random(
    four_sample_data,
):
    """Sequential mode reseeds each sample from its job_number, so the
    four samples must record four different np_draw/py_draw values."""
    result = evaluate_sequential(
        execute_fn=_draw_globals_fn, data=four_sample_data, pass_id=False
    )
    _, df = result.to_pandas()
    assert df["np_draw"].nunique() == len(df)
    assert df["py_draw"].nunique() == len(df)


def test_evaluate_multiprocessing_workers_see_distinct_random(
    four_sample_data,
):
    """Parallel workers fork from the parent's RNG state. Without the
    per-job reseed every worker would record the same np_draw."""
    result = evaluate_multiprocessing(
        execute_fn=_draw_globals_fn,
        data=four_sample_data,
        pass_id=False,
        nodes=4,
    )
    _, df = result.to_pandas()
    assert df["np_draw"].nunique() == len(df), (
        f"expected 4 distinct np draws, got {df['np_draw'].tolist()}"
    )
    assert df["py_draw"].nunique() == len(df), (
        f"expected 4 distinct py draws, got {df['py_draw'].tolist()}"
    )
