"""Regression tests for issue #234: optional tqdm progress bar plus
driver-side ``ExperimentData.progress_summary``."""

import sys
from unittest import mock

import pytest

from f3dasm import ExperimentData, ExperimentSample, create_sampler
from f3dasm._src.datagenerator import (
    _progress_bar,
    _progress_iter,
    evaluate_multiprocessing,
    evaluate_sequential,
)
from f3dasm.design import Domain

pytestmark = pytest.mark.smoke


def _square_fn(experiment_sample: ExperimentSample, **kwargs):
    x = experiment_sample.get("x")
    experiment_sample.store("y", x * x, to_disk=False)
    return experiment_sample


@pytest.fixture
def small_data():
    domain = Domain()
    domain.add_float(name="x", low=0.0, high=1.0)
    domain.add_output(name="y")
    data = ExperimentData(domain=domain)
    return create_sampler("random", seed=42).call(data=data, n_samples=4)


class TestProgressHelpers:
    def test_progress_bar_disabled_is_noop(self):
        with _progress_bar(total=10, enabled=False, desc="x") as update:
            update(3)  # must not error and must not require tqdm
            update(1)

    def test_progress_iter_disabled_returns_iterable(self):
        items = [1, 2, 3]
        wrapped = _progress_iter(items, total=3, enabled=False, desc="x")
        assert list(wrapped) == items

    def test_progress_bar_graceful_when_tqdm_missing(self, monkeypatch):
        # Simulate tqdm not being installed: blocking the import returns
        # ImportError and the helper must still yield a no-op update.
        monkeypatch.setitem(sys.modules, "tqdm", None)
        monkeypatch.setitem(sys.modules, "tqdm.auto", None)
        with _progress_bar(total=4, enabled=True, desc="x") as update:
            update(2)
            update(2)


def test_evaluate_sequential_progress_silent_without_tqdm(
    small_data, monkeypatch
):
    monkeypatch.setitem(sys.modules, "tqdm", None)
    monkeypatch.setitem(sys.modules, "tqdm.auto", None)
    # Must run cleanly with progress=True even when tqdm is unavailable.
    result = evaluate_sequential(
        execute_fn=_square_fn,
        data=small_data,
        pass_id=False,
        progress=True,
    )
    _, out = result.to_pandas()
    assert len(out) == 4


def test_evaluate_multiprocessing_progress_silent_without_tqdm(
    small_data, monkeypatch
):
    monkeypatch.setitem(sys.modules, "tqdm", None)
    monkeypatch.setitem(sys.modules, "tqdm.auto", None)
    result = evaluate_multiprocessing(
        execute_fn=_square_fn,
        data=small_data,
        pass_id=False,
        nodes=2,
        progress=True,
    )
    _, out = result.to_pandas()
    assert len(out) == 4


def test_progress_summary_reflects_status(small_data):
    """`progress_summary` is the cluster/MPI-friendly view that works
    regardless of evaluator."""
    summary = small_data.progress_summary()
    assert summary == {
        "open": 4,
        "in_progress": 0,
        "finished": 0,
        "error": 0,
    }

    # After evaluation the counts should shift to "finished".
    evaluated = evaluate_sequential(
        execute_fn=_square_fn, data=small_data, pass_id=False
    )
    summary = evaluated.progress_summary()
    assert summary == {
        "open": 0,
        "in_progress": 0,
        "finished": 4,
        "error": 0,
    }


def test_progress_summary_keys_are_stable(small_data):
    # Even when no samples are in a given state, the key must be present
    # so downstream code (driver loops) can rely on the schema.
    keys = set(small_data.progress_summary())
    assert keys == {"open", "in_progress", "finished", "error"}


def test_progress_bar_uses_tqdm_when_enabled():
    """When `progress=True` and tqdm is importable, the helper must
    actually emit a tqdm instance and tick it once per update."""
    bar_instance = mock.MagicMock()
    fake_tqdm = mock.MagicMock(return_value=bar_instance)
    with mock.patch.dict(
        sys.modules, {"tqdm.auto": mock.MagicMock(tqdm=fake_tqdm)}
    ):
        with _progress_bar(total=3, enabled=True, desc="x") as update:
            update(1)
            update(2)
    fake_tqdm.assert_called_once_with(total=3, desc="x", leave=False)
    assert bar_instance.update.call_count == 2
    bar_instance.close.assert_called_once()
