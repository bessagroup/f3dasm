"""Tests for SLURM executor script rendering."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from f3dasm._src.pipeline.executors.slurm import (
    SlurmExecutor,
    _get_next_dependency,
    _render_loop_block,
    _render_step_block,
    render_orchestrator_script,
    render_sbatch_script,
)
from f3dasm._src.pipeline.loop import Loop
from f3dasm._src.pipeline.pipeline import Pipeline, Step
from f3dasm._src.pipeline.resources import SlurmCluster, SlurmResources

pytestmark = pytest.mark.smoke


@pytest.fixture
def cluster():
    return SlurmCluster(
        partition="compute",
        account="proj123",
        env_setup=["module load python/3.11"],
        env_vars={"MY_VAR": "value"},
        runner="python",
    )


@pytest.fixture
def resources():
    return SlurmResources(
        time="01:00:00",
        mem="4G",
        cpus_per_task=2,
        extra_sbatch={"gres": "gpu:1"},
    )


class TestRenderSbatchScript:
    def test_basic_script(self, cluster, resources):
        job_dir = Path("/scratch/job1")
        step = Step(block=lambda: None, name="train", resources=resources)
        script = render_sbatch_script(
            step=step,
            cluster=cluster,
            pipeline_name="my_pipe",
            label="train",
            job_dir=job_dir,
            iteration=0,
        )
        assert "#!/bin/bash" in script
        assert "#SBATCH --job-name=train_my_pipe" in script
        assert "#SBATCH --time=01:00:00" in script
        assert "#SBATCH --mem=4G" in script
        assert "#SBATCH --cpus-per-task=2" in script
        assert "#SBATCH --partition=compute" in script
        assert "#SBATCH --account=proj123" in script
        assert "#SBATCH --gres=gpu:1" in script
        assert "module load python/3.11" in script
        assert 'export MY_VAR="value"' in script
        assert "--step=train" in script
        assert f"--job-dir={job_dir.as_posix()}" in script
        assert "--iteration=0" in script
        # Non-parallel: no array or job-number
        assert "--array" not in script
        assert "--job-number" not in script

    def test_parallel_script_no_array_directive(self, cluster):
        # Array size is supplied on the sbatch command line by the
        # orchestrator, so the per-step script must NOT bake one in.
        res = SlurmResources(max_array_size=100, max_concurrent=32)
        step = Step(
            block=lambda: None, name="run", resources=res, parallel=True
        )
        script = render_sbatch_script(
            step=step,
            cluster=cluster,
            pipeline_name="pipe",
            label="run",
            job_dir=Path("/scratch/job1"),
            iteration=0,
        )
        assert "#SBATCH --array" not in script
        assert "--job-number=$SLURM_ARRAY_TASK_ID" in script
        assert "%A_%a.out" in script

    def test_shell_variable_iteration(self, cluster):
        step = Step(block=lambda: None, name="run")
        script = render_sbatch_script(
            step=step,
            cluster=cluster,
            pipeline_name="pipe",
            label="run",
            job_dir=Path("/scratch/job1"),
            iteration="$F3DASM_ITERATION",
        )
        assert "--iteration=$F3DASM_ITERATION" in script

    def test_non_parallel_log_path(self, cluster):
        step = Step(block=lambda: None, name="run")
        script = render_sbatch_script(
            step=step,
            cluster=cluster,
            pipeline_name="pipe",
            label="run",
            job_dir=Path("/scratch/job1"),
            iteration=0,
        )
        assert "%j.out" in script


class TestGetNextDependency:
    def test_no_next_element(self):
        p = Pipeline(steps=[Step(block=lambda: None, name="a")])
        result = _get_next_dependency(p, 0, 1)
        assert result is None

    def test_next_is_step(self):
        p = Pipeline(
            steps=[
                Step(block=lambda: None, name="a"),
                Step(block=lambda: None, name="b", dependency="afterany"),
            ]
        )
        result = _get_next_dependency(p, 0, 2)
        assert result == "afterany"

    def test_next_is_loop(self):
        inner = Step(block=lambda: None, name="inner", dependency="afterany")
        p = Pipeline(
            steps=[
                Step(block=lambda: None, name="a"),
                Loop(n_iterations=2, steps=[inner]),
            ]
        )
        result = _get_next_dependency(p, 0, 2)
        assert result == "afterany"

    def test_next_is_empty_loop(self):
        p = Pipeline(
            steps=[
                Step(block=lambda: None, name="a"),
                Loop(n_iterations=2, steps=[]),
            ]
        )
        result = _get_next_dependency(p, 0, 2)
        assert result == "afterok"


class TestRenderStepBlock:
    def test_step_with_next(self, cluster):
        lines = []
        step = Step(block=lambda: None, name="a")
        p = Pipeline(steps=[step, Step(block=lambda: None, name="b")])
        _render_step_block(
            lines=lines,
            step=step,
            step_index=0,
            pipeline=p,
            cluster=cluster,
            script_paths={"a": "/scripts/a.sh"},
            total_steps=2,
        )
        text = "\n".join(lines)
        assert "sbatch" in text
        assert "STEP_COUNT=1" in text
        assert "exit 0" in text

    def test_last_step(self, cluster):
        lines = []
        step = Step(block=lambda: None, name="a")
        p = Pipeline(steps=[step])
        _render_step_block(
            lines=lines,
            step=step,
            step_index=0,
            pipeline=p,
            cluster=cluster,
            script_paths={"a": "/scripts/a.sh"},
            total_steps=1,
        )
        text = "\n".join(lines)
        assert "exit 0" in text
        assert "STEP_COUNT" not in text

    def test_parallel_step_resolves_array_at_submit(self, cluster):
        # A parallel step must (a) call count_open to determine
        # the array width, (b) sbatch with a runtime --array=
        # flag, and (c) handle the zero-open case by skipping
        # submission and resubmitting without a dependency.
        lines = []
        res = SlurmResources(max_array_size=900, max_concurrent=64)
        step = Step(
            block=lambda: None, name="run", parallel=True, resources=res
        )
        p = Pipeline(steps=[step, Step(block=lambda: None, name="post")])
        _render_step_block(
            lines=lines,
            step=step,
            step_index=0,
            pipeline=p,
            cluster=cluster,
            script_paths={"run": "/scripts/run.sh"},
            total_steps=2,
        )
        text = "\n".join(lines)
        assert "f3dasm.pipeline.count_open" in text
        assert "N_OPEN" in text
        assert "--array=0-${ARRAY_MAX}%64" in text
        assert "(N_OPEN < 900 ? N_OPEN : 900) - 1" in text
        # Skip + no-dep resubmit path for empty-open case
        assert 'JOB_ID=""' in text
        assert 'if [ -n "$JOB_ID" ]; then' in text


class TestRenderLoopBlock:
    def test_loop_block(self, cluster):
        lines = []
        inner = Step(block=lambda: None, name="train")
        loop = Loop(n_iterations=5, steps=[inner])
        p = Pipeline(steps=[loop])
        _render_loop_block(
            lines=lines,
            loop=loop,
            step_index=0,
            pipeline=p,
            cluster=cluster,
            script_paths={"loop0_train": "/scripts/loop0_train.sh"},
            total_steps=1,
        )
        text = "\n".join(lines)
        assert "5 iterations" in text
        assert "F3DASM_ITERATION" in text
        assert "LOOP_COUNT" in text
        assert "train" in text

    def test_loop_with_multiple_inner_steps(self, cluster):
        lines = []
        s1 = Step(block=lambda: None, name="gen")
        s2 = Step(block=lambda: None, name="post", dependency="afterany")
        loop = Loop(n_iterations=3, steps=[s1, s2])
        p = Pipeline(steps=[loop])
        _render_loop_block(
            lines=lines,
            loop=loop,
            step_index=0,
            pipeline=p,
            cluster=cluster,
            script_paths={
                "loop0_gen": "/scripts/loop0_gen.sh",
                "loop0_post": "/scripts/loop0_post.sh",
            },
            total_steps=1,
        )
        text = "\n".join(lines)
        assert "dependency=afterany" in text
        assert "gen" in text
        assert "post" in text

    def test_loop_with_parallel_inner_step(self, cluster):
        lines = []
        res = SlurmResources(max_array_size=900, max_concurrent=64)
        s1 = Step(block=lambda: None, name="gen", parallel=True, resources=res)
        s2 = Step(block=lambda: None, name="post")
        loop = Loop(n_iterations=3, steps=[s1, s2])
        p = Pipeline(steps=[loop])
        _render_loop_block(
            lines=lines,
            loop=loop,
            step_index=0,
            pipeline=p,
            cluster=cluster,
            script_paths={
                "loop0_gen": "/scripts/loop0_gen.sh",
                "loop0_post": "/scripts/loop0_post.sh",
            },
            total_steps=1,
        )
        text = "\n".join(lines)
        assert "f3dasm.pipeline.count_open" in text
        assert "--array=0-${ARRAY_MAX}%64" in text
        # PREV_JOB_ID is set/used conditionally for chaining
        assert 'if [ -n "$PREV_JOB_ID" ]; then' in text


class TestRenderOrchestratorScript:
    def test_basic_orchestrator(self, cluster):
        step = Step(block=lambda: None, name="create")
        p = Pipeline(name="test", steps=[step])
        res = SlurmResources(time="00:05:00", mem="1G")
        script = render_orchestrator_script(
            pipeline=p,
            cluster=cluster,
            orchestrator_resources=res,
            script_paths={"create": "/scripts/create.sh"},
            log_dir_path="/logs",
            job_dir=Path("/scratch/job1"),
        )
        assert "#!/bin/bash" in script
        assert "orchestrator_test" in script
        assert "STEP_COUNT=$1" in script
        assert "LOOP_COUNT=$2" in script
        assert "TOTAL_STEPS=1" in script
        assert 'JOB_DIR="/scratch/job1"' in script
        assert "Pipeline complete" in script

    def test_orchestrator_inherits_env_setup(self, cluster):
        # The orchestrator runs count_open before each parallel
        # sbatch, so it must source the cluster's env_setup.
        step = Step(block=lambda: None, name="create")
        p = Pipeline(name="test", steps=[step])
        res = SlurmResources(time="00:05:00", mem="1G")
        script = render_orchestrator_script(
            pipeline=p,
            cluster=cluster,
            orchestrator_resources=res,
            script_paths={"create": "/scripts/create.sh"},
            log_dir_path="/logs",
            job_dir=Path("/scratch/job1"),
        )
        assert "module load python/3.11" in script
        assert 'export MY_VAR="value"' in script


class TestSlurmExecutorGenerateScripts:
    def test_generate_scripts(self, cluster):
        step_a = Step(block=lambda: None, name="create")
        inner = Step(block=lambda: None, name="run")
        loop = Loop(n_iterations=3, steps=[inner])
        p = Pipeline(name="test", steps=[step_a, loop])

        executor = SlurmExecutor(cluster=cluster)
        scripts = executor.generate_scripts(
            pipeline=p, project_job="test_job", rootdir=Path("/scratch")
        )
        assert "create" in scripts
        assert "loop1_run" in scripts
        assert "orchestrator" in scripts
        assert "#!/bin/bash" in scripts["orchestrator"]


class TestSysPathSerialization:
    def test_sys_path_json_written(self, cluster, tmp_path):
        """SlurmExecutor.run() writes .sys_path.json next to .pipeline.pkl."""
        step = Step(block=lambda: None, name="train")
        p = Pipeline(name="test", steps=[step])
        executor = SlurmExecutor(cluster=cluster)

        mock_result = type(
            "Result", (), {"stdout": "Submitted batch job 12345"}
        )()
        with patch("subprocess.run", return_value=mock_result):
            job_id = executor.run(
                pipeline=p, project_job="myjob", rootdir=tmp_path
            )

        job_dir = tmp_path / job_id
        sys_path_file = job_dir / ".sys_path.json"
        assert sys_path_file.exists()

        paths = json.loads(sys_path_file.read_text())
        assert isinstance(paths, list)
        assert len(paths) > 0
        # All entries should be absolute paths (no empty strings
        # or relative paths).
        for p in paths:
            assert p, "empty string should have been resolved"
            assert Path(p).is_absolute(), f"expected absolute: {p}"

    def test_sys_path_no_duplicates(self, cluster, tmp_path):
        """Normalized sys.path should not contain duplicates."""
        step = Step(block=lambda: None, name="train")
        p = Pipeline(name="test", steps=[step])
        executor = SlurmExecutor(cluster=cluster)

        mock_result = type(
            "Result", (), {"stdout": "Submitted batch job 12345"}
        )()
        with patch("subprocess.run", return_value=mock_result):
            executor.run(pipeline=p, project_job="myjob", rootdir=tmp_path)

        paths = json.loads((tmp_path / "myjob" / ".sys_path.json").read_text())
        assert len(paths) == len(set(paths))
