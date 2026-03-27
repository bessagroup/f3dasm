"""Tests for SLURM executor script rendering."""

from pathlib import Path

import pytest

from f3dasm._src.pipeline.executors.slurm import (
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
        step = Step(block=lambda: None, name="train", resources=resources)
        script = render_sbatch_script(
            step=step,
            cluster=cluster,
            pipeline_name="my_pipe",
            label="train",
            job_dir=Path("/scratch/job1"),
            n_jobs=None,
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
        assert "--job-dir=/scratch/job1" in script
        assert "--iteration=0" in script
        # Non-parallel: no array or job-number
        assert "--array" not in script
        assert "--job-number" not in script

    def test_parallel_script(self, cluster):
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
            n_jobs=50,
            iteration=0,
        )
        assert "#SBATCH --array=0-50%32" in script
        assert "--job-number=$SLURM_ARRAY_TASK_ID" in script
        assert "%A_%a.out" in script

    def test_parallel_array_capped(self, cluster):
        res = SlurmResources(max_array_size=10, max_concurrent=5)
        step = Step(
            block=lambda: None, name="run", resources=res, parallel=True
        )
        script = render_sbatch_script(
            step=step,
            cluster=cluster,
            pipeline_name="pipe",
            label="run",
            job_dir=Path("/scratch/job1"),
            n_jobs=100,
            iteration=0,
        )
        assert "#SBATCH --array=0-10%5" in script

    def test_shell_variable_iteration(self, cluster):
        step = Step(block=lambda: None, name="run")
        script = render_sbatch_script(
            step=step,
            cluster=cluster,
            pipeline_name="pipe",
            label="run",
            job_dir=Path("/scratch/job1"),
            n_jobs=None,
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
            n_jobs=None,
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
    def test_step_with_next(self):
        lines = []
        step = Step(block=lambda: None, name="a")
        p = Pipeline(
            steps=[step, Step(block=lambda: None, name="b")]
        )
        _render_step_block(
            lines=lines,
            step=step,
            step_index=0,
            pipeline=p,
            script_paths={"a": "/scripts/a.sh"},
            total_steps=2,
        )
        text = "\n".join(lines)
        assert "sbatch" in text
        assert "STEP_COUNT=1" in text
        assert "exit 0" in text

    def test_last_step(self):
        lines = []
        step = Step(block=lambda: None, name="a")
        p = Pipeline(steps=[step])
        _render_step_block(
            lines=lines,
            step=step,
            step_index=0,
            pipeline=p,
            script_paths={"a": "/scripts/a.sh"},
            total_steps=1,
        )
        text = "\n".join(lines)
        assert "exit 0" in text
        assert "STEP_COUNT" not in text


class TestRenderLoopBlock:
    def test_loop_block(self):
        lines = []
        inner = Step(block=lambda: None, name="train")
        loop = Loop(n_iterations=5, steps=[inner])
        p = Pipeline(steps=[loop])
        _render_loop_block(
            lines=lines,
            loop=loop,
            step_index=0,
            pipeline=p,
            script_paths={"loop0_train": "/scripts/loop0_train.sh"},
            total_steps=1,
        )
        text = "\n".join(lines)
        assert "5 iterations" in text
        assert "F3DASM_ITERATION" in text
        assert "LOOP_COUNT" in text
        assert "train" in text

    def test_loop_with_multiple_inner_steps(self):
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
        )
        assert "#!/bin/bash" in script
        assert "orchestrator_test" in script
        assert "STEP_COUNT=$1" in script
        assert "LOOP_COUNT=$2" in script
        assert "TOTAL_STEPS=1" in script
        assert "Pipeline complete" in script


class TestSlurmExecutorGenerateScripts:
    def test_generate_scripts(self, cluster):
        from f3dasm._src.pipeline.executors.slurm import SlurmExecutor

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
