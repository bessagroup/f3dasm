"""Tests for Pipeline and Step dataclasses."""

from unittest.mock import MagicMock, patch

import pytest

from f3dasm._src.pipeline.loop import Loop
from f3dasm._src.pipeline.pipeline import VALID_DEPENDENCIES, Pipeline, Step
from f3dasm._src.pipeline.resources import SlurmCluster, SlurmResources

pytestmark = pytest.mark.smoke


# ---- Step tests ----


class TestStep:
    def test_default_step(self):
        step = Step(block=lambda: None)
        assert step.name == ""
        assert step.parallel is False
        assert step.dependency == "afterok"
        assert step.array_jobs is None
        assert step.project_dir == "."
        assert step.kwargs == {}
        assert isinstance(step.resources, SlurmResources)

    def test_step_with_custom_values(self):
        res = SlurmResources(time="02:00:00", mem="8G")
        step = Step(
            block=lambda: None,
            name="train",
            parallel=True,
            resources=res,
            dependency="afterany",
            array_jobs=10,
            project_dir="subdir",
            kwargs={"lr": 0.01},
        )
        assert step.name == "train"
        assert step.parallel is True
        assert step.resources.time == "02:00:00"
        assert step.dependency == "afterany"
        assert step.array_jobs == 10
        assert step.project_dir == "subdir"
        assert step.kwargs == {"lr": 0.01}

    def test_step_invalid_dependency(self):
        with pytest.raises(ValueError, match="Invalid dependency"):
            Step(block=lambda: None, dependency="invalid")

    @pytest.mark.parametrize("dep", VALID_DEPENDENCIES)
    def test_step_valid_dependencies(self, dep):
        step = Step(block=lambda: None, dependency=dep)
        assert step.dependency == dep


# ---- Pipeline tests ----


class TestPipeline:
    def test_empty_pipeline(self):
        p = Pipeline(name="empty")
        assert p.name == "empty"
        assert p.steps == []
        assert p.orchestrator_resources is None

    def test_flatten_steps_only(self):
        steps = [
            Step(block=lambda: None, name="a"),
            Step(block=lambda: None, name="b"),
        ]
        p = Pipeline(name="test", steps=steps)
        flat = p._flatten()
        assert len(flat) == 2
        assert flat[0] == (steps[0], 0, 1)
        assert flat[1] == (steps[1], 0, 1)

    def test_flatten_with_loop(self):
        step_a = Step(block=lambda: None, name="a")
        inner_step = Step(block=lambda: None, name="inner")
        loop = Loop(n_iterations=3, steps=[inner_step])
        p = Pipeline(name="test", steps=[step_a, loop])
        flat = p._flatten()
        assert len(flat) == 4  # 1 step + 3 iterations
        assert flat[0] == (step_a, 0, 1)
        assert flat[1] == (inner_step, 0, 3)
        assert flat[2] == (inner_step, 1, 3)
        assert flat[3] == (inner_step, 2, 3)

    def test_flatten_loop_with_multiple_steps(self):
        s1 = Step(block=lambda: None, name="s1")
        s2 = Step(block=lambda: None, name="s2")
        loop = Loop(n_iterations=2, steps=[s1, s2])
        p = Pipeline(name="test", steps=[loop])
        flat = p._flatten()
        assert len(flat) == 4
        assert flat[0] == (s1, 0, 2)
        assert flat[1] == (s2, 0, 2)
        assert flat[2] == (s1, 1, 2)
        assert flat[3] == (s2, 1, 2)

    def test_flatten_empty_loop(self):
        loop = Loop(n_iterations=3, steps=[])
        p = Pipeline(name="test", steps=[loop])
        flat = p._flatten()
        assert flat == []

    def test_from_step_by_index(self):
        steps = [
            Step(block=lambda: None, name="a"),
            Step(block=lambda: None, name="b"),
            Step(block=lambda: None, name="c"),
        ]
        p = Pipeline(name="test", steps=steps)
        sub = p.from_step(1)
        assert len(sub.steps) == 2
        assert sub.steps[0].name == "b"
        assert sub.steps[1].name == "c"
        assert sub.name == "test"

    def test_from_step_by_name(self):
        steps = [
            Step(block=lambda: None, name="a"),
            Step(block=lambda: None, name="b"),
            Step(block=lambda: None, name="c"),
        ]
        p = Pipeline(name="test", steps=steps)
        sub = p.from_step("b")
        assert len(sub.steps) == 2
        assert sub.steps[0].name == "b"

    def test_from_step_name_in_loop(self):
        step_a = Step(block=lambda: None, name="a")
        inner = Step(block=lambda: None, name="inner")
        loop = Loop(n_iterations=3, steps=[inner])
        step_c = Step(block=lambda: None, name="c")
        p = Pipeline(name="test", steps=[step_a, loop, step_c])
        sub = p.from_step("inner")
        # Should include the loop and everything after it
        assert len(sub.steps) == 2
        assert isinstance(sub.steps[0], Loop)
        assert sub.steps[1].name == "c"

    def test_from_step_name_not_found(self):
        p = Pipeline(name="test", steps=[Step(block=lambda: None, name="a")])
        with pytest.raises(ValueError, match="not found"):
            p.from_step("nonexistent")

    def test_run_invalid_mode(self):
        p = Pipeline(name="test", steps=[])
        with pytest.raises(ValueError, match="Unknown mode"):
            p.run(mode="invalid")

    def test_run_slurm_without_cluster(self):
        p = Pipeline(name="test", steps=[])
        with pytest.raises(ValueError, match="SlurmCluster must be provided"):
            p.run(mode="slurm")


# ---- Loop tests ----


class TestLoop:
    def test_default_loop(self):
        loop = Loop()
        assert loop.n_iterations == 1
        assert loop.steps == []

    def test_loop_with_steps(self):
        s = Step(block=lambda: None, name="s")
        loop = Loop(n_iterations=5, steps=[s])
        assert loop.n_iterations == 5
        assert len(loop.steps) == 1
        assert loop.steps[0].name == "s"


# ---- Pipeline.run() tests ----


class TestPipelineRun:
    def test_run_local_mode(self):
        p = Pipeline(name="test", steps=[])
        with patch(
            "f3dasm._src.pipeline.executors.local.LocalExecutor"
        ) as mock_cls:
            mock_executor = MagicMock()
            mock_cls.return_value = mock_executor
            p.run(mode="local")
            mock_executor.run.assert_called_once()

    def test_run_slurm_mode(self):
        p = Pipeline(name="test", steps=[])
        cluster = SlurmCluster(partition="gpu", account="test")
        with patch(
            "f3dasm._src.pipeline.executors.slurm.SlurmExecutor"
        ) as mock_cls:
            mock_executor = MagicMock()
            mock_cls.return_value = mock_executor
            p.run(mode="slurm", cluster=cluster)
            mock_executor.run.assert_called_once()

    def test_generate_scripts(self, tmp_path):
        p = Pipeline(name="test", steps=[])
        cluster = SlurmCluster(partition="gpu", account="test")
        with patch(
            "f3dasm._src.pipeline.executors.slurm.SlurmExecutor"
        ) as mock_cls:
            mock_executor = MagicMock()
            mock_cls.return_value = mock_executor
            p.generate_scripts(cluster=cluster, rootdir=tmp_path)
            mock_executor.generate_scripts.assert_called_once()
