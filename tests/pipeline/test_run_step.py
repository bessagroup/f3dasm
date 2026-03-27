"""Tests for the run_step CLI entry point."""

import pytest

from f3dasm._src.pipeline.loop import Loop
from f3dasm._src.pipeline.pipeline import Pipeline, Step
from f3dasm._src.pipeline.run_step import _execute_step, _find_step

pytestmark = pytest.mark.smoke


class TestFindStep:
    def test_find_top_level_step(self):
        step = Step(block=lambda: None, name="train")
        p = Pipeline(steps=[step])
        assert _find_step(p, "train") is step

    def test_find_step_in_loop(self):
        inner = Step(block=lambda: None, name="inner")
        loop = Loop(n_iterations=3, steps=[inner])
        p = Pipeline(steps=[loop])
        assert _find_step(p, "inner") is inner

    def test_step_not_found(self):
        p = Pipeline(
            steps=[Step(block=lambda: None, name="a")]
        )
        assert _find_step(p, "nonexistent") is None

    def test_empty_pipeline(self):
        p = Pipeline(steps=[])
        assert _find_step(p, "anything") is None


class TestExecuteStep:
    def test_callable_block(self, tmp_path):
        called = []

        def my_fn(project_dir, **kwargs):
            called.append(True)

        step = Step(block=my_fn, name="create")
        _execute_step(step=step, run_dir=tmp_path, job_number=None)
        assert len(called) == 1

    def test_callable_block_with_kwargs(self, tmp_path):
        received = {}

        def my_fn(project_dir, **kwargs):
            received.update(kwargs)

        step = Step(block=my_fn, name="create", kwargs={"x": 42})
        _execute_step(step=step, run_dir=tmp_path, job_number=None)
        assert received == {"x": 42}

    def test_unsupported_block_type(self, tmp_path):
        step = Step(block=123, name="bad")
        with pytest.raises(TypeError, match="unsupported block type"):
            _execute_step(step=step, run_dir=tmp_path, job_number=None)
