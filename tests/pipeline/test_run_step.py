"""Tests for the run_step CLI entry point."""

import json
import sys

import cloudpickle
import pytest

from f3dasm._src.pipeline.loop import Loop
from f3dasm._src.pipeline.pipeline import Pipeline, Step
from f3dasm._src.pipeline.run_step import _execute_step, _find_step, main

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
        p = Pipeline(steps=[Step(block=lambda: None, name="a")])
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


class TestSysPathRestoration:
    def _make_job_dir(self, tmp_path):
        """Create a job directory with a pickled pipeline.

        The step writes a marker file so we can verify it ran
        (the callable is unpickled into a new object, so we
        can't track calls via a shared list).
        """
        marker = tmp_path / "marker.txt"

        def my_fn(project_dir, **kwargs):
            marker.write_text("executed")

        step = Step(block=my_fn, name="create")
        p = Pipeline(name="test", steps=[step])

        job_dir = tmp_path / "job"
        job_dir.mkdir()
        with open(job_dir / ".pipeline.pkl", "wb") as f:
            cloudpickle.dump(p, f)
        return job_dir, marker

    def test_sys_path_restored_from_json(self, tmp_path):
        """Paths from .sys_path.json are prepended to sys.path."""
        job_dir, marker = self._make_job_dir(tmp_path)

        fake_dir = str(tmp_path / "user_project")
        with open(job_dir / ".sys_path.json", "w") as f:
            json.dump([fake_dir], f)

        main(["--step=create", f"--job-dir={job_dir}"])

        assert fake_dir in sys.path
        assert marker.exists()

        # Cleanup
        sys.path.remove(fake_dir)

    def test_ordering_preserved(self, tmp_path):
        """Stored paths should appear in the same order in sys.path."""
        job_dir, _ = self._make_job_dir(tmp_path)

        paths = [
            str(tmp_path / "aaa"),
            str(tmp_path / "bbb"),
            str(tmp_path / "ccc"),
        ]
        with open(job_dir / ".sys_path.json", "w") as f:
            json.dump(paths, f)

        main(["--step=create", f"--job-dir={job_dir}"])

        indices = [sys.path.index(p) for p in paths]
        assert indices == sorted(indices), "paths should preserve order"

        # Cleanup
        for p in paths:
            sys.path.remove(p)

    def test_no_duplicates_added(self, tmp_path):
        """Paths already in sys.path should not be duplicated."""
        job_dir, _ = self._make_job_dir(tmp_path)

        existing = sys.path[0]
        with open(job_dir / ".sys_path.json", "w") as f:
            json.dump([existing], f)

        count_before = sys.path.count(existing)
        main(["--step=create", f"--job-dir={job_dir}"])
        count_after = sys.path.count(existing)

        assert count_after == count_before

    def test_backward_compat_no_json(self, tmp_path):
        """Without .sys_path.json the pipeline still runs."""
        job_dir, marker = self._make_job_dir(tmp_path)

        # No .sys_path.json written
        main(["--step=create", f"--job-dir={job_dir}"])
        assert marker.exists()
