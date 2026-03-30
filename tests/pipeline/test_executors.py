"""Tests for pipeline executors."""

import pytest

from f3dasm._src.pipeline.executors.base import Executor
from f3dasm._src.pipeline.executors.local import (
    LocalExecutor,
    _run_step_locally,
)
from f3dasm._src.pipeline.pipeline import Pipeline, Step

pytestmark = pytest.mark.smoke


class TestExecutorABC:
    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            Executor()


class TestLocalExecutor:
    def test_default_parallel_mode(self):
        executor = LocalExecutor()
        assert executor.parallel_mode == "cluster"

    def test_custom_parallel_mode(self):
        executor = LocalExecutor(parallel_mode="sequential")
        assert executor.parallel_mode == "sequential"

    def test_run_empty_pipeline(self, tmp_path):
        executor = LocalExecutor()
        p = Pipeline(name="empty", steps=[])
        job_id = executor.run(p, project_job="test_job", rootdir=tmp_path)
        assert job_id == "test_job"
        assert (tmp_path / "test_job").is_dir()

    def test_run_generates_job_id(self, tmp_path):
        executor = LocalExecutor()
        p = Pipeline(name="empty", steps=[])
        job_id = executor.run(p, rootdir=tmp_path)
        assert job_id.isdigit()

    def test_run_with_callable_step(self, tmp_path):
        called = []

        def my_block(project_dir, **kwargs):
            called.append(str(project_dir))

        p = Pipeline(
            name="test",
            steps=[Step(block=my_block, name="create")],
        )
        executor = LocalExecutor()
        job_id = executor.run(p, project_job="run1", rootdir=tmp_path)
        assert job_id == "run1"
        assert len(called) == 1
        assert "run1" in called[0]

    def test_run_callable_with_kwargs(self, tmp_path):
        received_kwargs = {}

        def my_block(project_dir, **kwargs):
            received_kwargs.update(kwargs)

        p = Pipeline(
            name="test",
            steps=[
                Step(block=my_block, name="create", kwargs={"lr": 0.01}),
            ],
        )
        executor = LocalExecutor()
        executor.run(p, project_job="run1", rootdir=tmp_path)
        assert received_kwargs == {"lr": 0.01}

    def test_run_creates_step_project_dir(self, tmp_path):
        def my_block(project_dir, **kwargs):
            pass

        p = Pipeline(
            name="test",
            steps=[
                Step(block=my_block, name="create", project_dir="sub/dir"),
            ],
        )
        executor = LocalExecutor()
        executor.run(p, project_job="run1", rootdir=tmp_path)
        assert (tmp_path / "run1" / "sub" / "dir").is_dir()


class TestRunStepLocally:
    def test_unsupported_block_type(self, tmp_path):
        step = Step(block=42, name="bad_block")
        with pytest.raises(TypeError, match="unsupported block type"):
            _run_step_locally(step=step, run_dir=tmp_path)
