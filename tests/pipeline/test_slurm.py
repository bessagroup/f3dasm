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
        # --ntasks is always emitted (default 1), before --cpus-per-task.
        assert "#SBATCH --ntasks=1" in script
        assert "#SBATCH --cpus-per-task=2" in script
        assert "#SBATCH --nodes=1" in script
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

    def test_wave_step_passes_wave_flag(self, cluster):
        res = SlurmResources(max_array_size=100, max_jobs_per_task=1)
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
        assert "--wave=${F3DASM_WAVE:-0}" in script

    def test_opted_out_step_has_no_wave_flag(self, cluster):
        res = SlurmResources(max_array_size=100, max_jobs_per_task=None)
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
        assert "--wave" not in script


class TestWaveRendering:
    def _parallel_step(self, max_jobs_per_task):
        res = SlurmResources(
            max_array_size=100,
            max_concurrent=32,
            max_jobs_per_task=max_jobs_per_task,
        )
        return Step(
            block=lambda: None, name="run", resources=res, parallel=True
        )

    def test_wave_step_block(self, cluster):
        lines = []
        step = self._parallel_step(max_jobs_per_task=2)
        pipeline = Pipeline(name="pipe", steps=[step])
        _render_step_block(
            lines=lines,
            step=step,
            step_index=0,
            pipeline=pipeline,
            cluster=cluster,
            script_paths={"run": "/scripts/run.sh"},
            total_steps=1,
        )
        text = "\n".join(lines)
        # Window = max_array_size * max_jobs_per_task = 200.
        assert "REM=$(( N_OPEN - WAVE_COUNT * 200 ))" in text
        assert "N_TASKS=$(( (REM + 2 - 1) / 2 ))" in text
        assert "export F3DASM_WAVE=$WAVE_COUNT" in text
        assert "--export=ALL" in text
        assert (
            'if [ $(( (WAVE_COUNT + 1) * 200 )) -lt "$N_OPEN" ]; then' in text
        )
        assert (
            "sbatch --dependency=afterany:$JOB_ID"
            ' "$SELF" $STEP_COUNT $LOOP_COUNT 0 $((WAVE_COUNT + 1))' in text
        )

    def test_opted_out_step_block_has_no_wave_logic(self, cluster):
        lines = []
        step = self._parallel_step(max_jobs_per_task=None)
        pipeline = Pipeline(name="pipe", steps=[step])
        _render_step_block(
            lines=lines,
            step=step,
            step_index=0,
            pipeline=pipeline,
            cluster=cluster,
            script_paths={"run": "/scripts/run.sh"},
            total_steps=1,
        )
        text = "\n".join(lines)
        assert "WAVE_COUNT" not in text
        assert "F3DASM_WAVE" not in text

    def test_orchestrator_wave_counter_only_when_used(self, cluster):
        wave_pipeline = Pipeline(
            name="pipe", steps=[self._parallel_step(max_jobs_per_task=1)]
        )
        opted_out = Pipeline(
            name="pipe", steps=[self._parallel_step(max_jobs_per_task=None)]
        )
        kwargs = dict(
            cluster=cluster,
            orchestrator_resources=SlurmResources(),
            script_paths={"run": "/scripts/run.sh"},
            log_dir_path="/logs",
            job_dir=Path("/scratch/job1"),
        )
        with_waves = render_orchestrator_script(
            pipeline=wave_pipeline, **kwargs
        )
        without = render_orchestrator_script(pipeline=opted_out, **kwargs)
        assert "WAVE_COUNT=${4:-0}" in with_waves
        assert "WAVE_COUNT" not in without

    def test_loop_inner_wave_resubmits_same_inner_step(self, cluster):
        lines = []
        inner = self._parallel_step(max_jobs_per_task=1)
        loop = Loop(n_iterations=2, steps=[inner])
        pipeline = Pipeline(name="pipe", steps=[loop])
        _render_loop_block(
            lines=lines,
            loop=loop,
            step_index=0,
            pipeline=pipeline,
            cluster=cluster,
            script_paths={"loop0_run": "/scripts/loop0_run.sh"},
            total_steps=1,
        )
        text = "\n".join(lines)
        assert (
            "sbatch --dependency=afterany:$JOB_ID"
            ' "$SELF" $STEP_COUNT $LOOP_COUNT 0 $((WAVE_COUNT + 1))' in text
        )
        # Advancing to the next iteration resets the wave counter by
        # omitting the fourth argument.
        assert 'sbatch "$SELF" $STEP_COUNT $((LOOP_COUNT + 1)) 0' in text


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

    def test_last_step_resubmits_final_marker_run(self, cluster):
        # The last step must resubmit the orchestrator one final
        # time (afterok), past TOTAL_STEPS, so that run skips the
        # while loop and prints the completion marker. Without it
        # the marker is unreachable for Step-final pipelines.
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
        assert "STEP_COUNT=1" in text
        assert (
            'sbatch --dependency=afterok:$JOB_ID "$SELF"'
            " $STEP_COUNT $LOOP_COUNT" in text
        )
        assert "exit 0" in text

    def test_parallel_step_resolves_array_at_submit(self, cluster):
        # An opted-out (max_jobs_per_task=None) parallel step must
        # (a) call count_open to determine the array width,
        # (b) sbatch with a runtime --array= flag, and (c) handle
        # the zero-open case by skipping submission and
        # resubmitting without a dependency.
        lines = []
        res = SlurmResources(
            max_array_size=900, max_concurrent=64, max_jobs_per_task=None
        )
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
        # A skipped parallel inner step empties JOB_ID and resubmits
        # without a dependency (the gen-then-post chaining is carried by
        # the per-inner-step SELF resubmission, not an in-wake PREV_JOB_ID).
        assert 'JOB_ID=""' in text
        assert 'if [ -n "$JOB_ID" ]; then' in text

    def test_parallel_inner_step_counts_open_after_predecessor(self, cluster):
        # Regression: a parallel inner step's count_open must run in a
        # *separate* wake from (and gated on) its non-parallel predecessor,
        # so the array width reflects what the predecessor just wrote --
        # not the previous iteration's residual job statuses.
        res = SlurmResources(max_array_size=900, max_concurrent=64)
        sub = Step(block=lambda: None, name="subsample", dependency="afterok")
        run = Step(
            block=lambda: None,
            name="run",
            parallel=True,
            resources=res,
            dependency="afterok",
        )
        post = Step(block=lambda: None, name="post", dependency="afterany")
        loop = Loop(n_iterations=4, steps=[sub, run, post])
        p = Pipeline(steps=[loop])
        lines = []
        _render_loop_block(
            lines=lines,
            loop=loop,
            step_index=0,
            pipeline=p,
            cluster=cluster,
            script_paths={
                "loop0_subsample": "/s/sub.sh",
                "loop0_run": "/s/run.sh",
                "loop0_post": "/s/post.sh",
            },
            total_steps=1,
        )
        text = "\n".join(lines)
        assert 'if [ "$INNER_COUNT" -eq 0 ]' in text
        assert 'elif [ "$INNER_COUNT" -eq 1 ]' in text
        assert 'elif [ "$INNER_COUNT" -eq 2 ]' in text
        seg0 = text[
            text.index('if [ "$INNER_COUNT" -eq 0 ]') : text.index(
                'elif [ "$INNER_COUNT" -eq 1 ]'
            )
        ]
        seg1 = text[
            text.index('elif [ "$INNER_COUNT" -eq 1 ]') : text.index(
                'elif [ "$INNER_COUNT" -eq 2 ]'
            )
        ]
        # subsample wake submits the script but never counts open ...
        assert "/s/sub.sh" in seg0
        assert "count_open" not in seg0
        # ... and run's count_open lives only in the next (gated) wake.
        assert "count_open" in seg1
        assert "/s/run.sh" in seg1

    def test_inner_steps_thread_inner_count(self, cluster):
        # Each inner step resubmits the orchestrator for the next inner
        # index (gated by the *next* step's dependency); the last inner
        # step advances the iteration and resets INNER_COUNT to 0.
        sub = Step(block=lambda: None, name="subsample", dependency="afterok")
        run = Step(block=lambda: None, name="run", dependency="afterok")
        post = Step(block=lambda: None, name="post", dependency="afterany")
        loop = Loop(n_iterations=4, steps=[sub, run, post])
        p = Pipeline(steps=[loop])
        lines = []
        _render_loop_block(
            lines=lines,
            loop=loop,
            step_index=0,
            pipeline=p,
            cluster=cluster,
            script_paths={
                "loop0_subsample": "/s/sub.sh",
                "loop0_run": "/s/run.sh",
                "loop0_post": "/s/post.sh",
            },
            total_steps=1,
        )
        text = "\n".join(lines)
        # subsample -> run (run.dependency == afterok), next inner index 1
        assert (
            'sbatch --dependency=afterok:$JOB_ID "$SELF"'
            " $STEP_COUNT $LOOP_COUNT 1" in text
        )
        # run -> post (post.dependency == afterany), next inner index 2
        assert (
            'sbatch --dependency=afterany:$JOB_ID "$SELF"'
            " $STEP_COUNT $LOOP_COUNT 2" in text
        )
        # post (last) -> next iteration (steps[0].dependency == afterok),
        # LOOP_COUNT + 1 and INNER_COUNT reset to 0
        assert (
            'sbatch --dependency=afterok:$JOB_ID "$SELF"'
            " $STEP_COUNT $((LOOP_COUNT + 1)) 0" in text
        )
        assert "INNER_COUNT=0" in text

    def test_empty_loop_advances_without_submitting(self, cluster):
        loop = Loop(n_iterations=3, steps=[])
        p = Pipeline(steps=[loop, Step(block=lambda: None, name="after")])
        lines = []
        _render_loop_block(
            lines=lines,
            loop=loop,
            step_index=0,
            pipeline=p,
            cluster=cluster,
            script_paths={},
            total_steps=2,
        )
        text = "\n".join(lines)
        assert 'sbatch "$SELF" $STEP_COUNT $LOOP_COUNT 0' in text
        assert "STEP_COUNT=1" in text


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
        assert "INNER_COUNT=${3:-0}" in script
        assert "TOTAL_STEPS=1" in script
        assert 'JOB_DIR="/scratch/job1"' in script
        assert "Pipeline complete" in script

    def test_step_final_pipeline_reaches_marker(self, cluster):
        # A pipeline ending in a plain Step (no trailing Loop)
        # must still produce a final orchestrator run that prints
        # the completion marker: the last step block resubmits
        # with STEP_COUNT == TOTAL_STEPS, which skips the while
        # loop and falls through to the echo.
        steps = [
            Step(block=lambda: None, name="create"),
            Step(block=lambda: None, name="report"),
        ]
        p = Pipeline(name="test", steps=steps)
        res = SlurmResources(time="00:05:00", mem="1G")
        script = render_orchestrator_script(
            pipeline=p,
            cluster=cluster,
            orchestrator_resources=res,
            script_paths={
                "create": "/scripts/create.sh",
                "report": "/scripts/report.sh",
            },
            log_dir_path="/logs",
            job_dir=Path("/scratch/job1"),
        )
        assert "TOTAL_STEPS=2" in script
        # The report block (last element) advances past TOTAL_STEPS
        # and resubmits the orchestrator for the marker run.
        assert "STEP_COUNT=2" in script
        assert (
            'sbatch --dependency=afterok:$JOB_ID "$SELF"'
            " $STEP_COUNT $LOOP_COUNT" in script
        )
        assert 'echo "Pipeline complete."' in script

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


class TestPerCpuMemory:
    """Per-CPU memory accounting for DelftBlue-style sites (issue #352)."""

    def _orchestrator(self, cluster, res):
        step = Step(block=lambda: None, name="create")
        p = Pipeline(name="test", steps=[step])
        return render_orchestrator_script(
            pipeline=p,
            cluster=cluster,
            orchestrator_resources=res,
            script_paths={"create": "/scripts/create.sh"},
            log_dir_path="/logs",
            job_dir=Path("/scratch/job1"),
        )

    def _step(self, cluster, res):
        step = Step(block=lambda: None, name="train", resources=res)
        return render_sbatch_script(
            step=step,
            cluster=cluster,
            pipeline_name="pipe",
            label="train",
            job_dir=Path("/scratch/job1"),
            iteration=0,
        )

    def test_step_mem_per_cpu_replaces_mem(self, cluster):
        # When mem_per_cpu is set, emit --mem-per-cpu and NOT --mem
        # (SLURM treats the two as mutually exclusive -> fatal).
        res = SlurmResources(mem="8G", mem_per_cpu="3968M")
        script = self._step(cluster, res)
        assert "#SBATCH --mem-per-cpu=3968M" in script
        assert "#SBATCH --mem=" not in script

    def test_step_default_uses_mem(self, cluster):
        # Without mem_per_cpu, --mem is used and --mem-per-cpu absent.
        script = self._step(cluster, SlurmResources(mem="8G"))
        assert "#SBATCH --mem=8G" in script
        assert "#SBATCH --mem-per-cpu" not in script

    def test_step_ntasks_custom_value(self, cluster):
        script = self._step(cluster, SlurmResources(ntasks=4))
        assert "#SBATCH --ntasks=4" in script

    def test_step_nodes_omitted_when_mem_per_cpu_and_default_nodes(
        self, cluster
    ):
        # mem_per_cpu set AND nodes at default 1 -> --nodes dropped.
        res = SlurmResources(mem_per_cpu="3968M", nodes=1)
        script = self._step(cluster, res)
        assert "#SBATCH --nodes" not in script

    def test_step_nodes_emitted_when_explicit_multinode(self, cluster):
        # An explicit nodes > 1 is never silently dropped.
        res = SlurmResources(mem_per_cpu="3968M", nodes=2)
        script = self._step(cluster, res)
        assert "#SBATCH --nodes=2" in script

    def test_step_nodes_emitted_without_mem_per_cpu(self, cluster):
        # Omission only applies under mem_per_cpu; the default path
        # keeps --nodes=1.
        script = self._step(cluster, SlurmResources(nodes=1))
        assert "#SBATCH --nodes=1" in script

    def test_orchestrator_mem_per_cpu_replaces_mem(self, cluster):
        # The orchestrator header obeys the same rules, so a
        # DelftBlue pipeline can make its orchestrator submittable
        # via orchestrator_resources.
        res = SlurmResources(time="00:10:00", mem="1G", mem_per_cpu="1024M")
        script = self._orchestrator(cluster, res)
        assert "#SBATCH --mem-per-cpu=1024M" in script
        assert "#SBATCH --mem=" not in script
        assert "#SBATCH --ntasks=1" in script
        assert "#SBATCH --nodes" not in script

    def test_orchestrator_default_uses_mem(self, cluster):
        res = SlurmResources(time="00:10:00", mem="1G")
        script = self._orchestrator(cluster, res)
        assert "#SBATCH --mem=1G" in script
        assert "#SBATCH --mem-per-cpu" not in script
        assert "#SBATCH --ntasks=1" in script

    def test_cluster_cap_fits_leaves_cpus_unchanged(self):
        # mem within one core's cap: --mem-per-cpu = mem, cpus as-is.
        cluster = SlurmCluster(partition="compute", mem_per_cpu="3968M")
        script = self._step(cluster, SlurmResources(mem="2G", cpus_per_task=1))
        assert "#SBATCH --mem-per-cpu=2048M" in script
        assert "#SBATCH --cpus-per-task=1" in script
        assert "#SBATCH --mem=" not in script

    def test_cluster_cap_bumps_cpus_when_mem_exceeds_cap(self):
        # 8G on 1 core exceeds the 3968M cap -> bump to ceil(8192/3968)
        # = 3 cores, each ceil(8192/3)=2731M.
        cluster = SlurmCluster(partition="compute", mem_per_cpu="3968M")
        script = self._step(cluster, SlurmResources(mem="8G", cpus_per_task=1))
        assert "#SBATCH --cpus-per-task=3" in script
        assert "#SBATCH --mem-per-cpu=2731M" in script
        assert "#SBATCH --mem=" not in script

    def test_cluster_cap_divides_per_node_mem_across_declared_cpus(self):
        # A multi-core step's per-node mem is divided, NOT reused
        # verbatim: 32G / 9 cores (bumped from 4) = 3641M/core, so the
        # total stays ~32G rather than 32G*cpus.
        cluster = SlurmCluster(partition="compute", mem_per_cpu="3968M")
        script = self._step(
            cluster, SlurmResources(mem="32G", cpus_per_task=4)
        )
        assert "#SBATCH --cpus-per-task=9" in script
        assert "#SBATCH --mem-per-cpu=3641M" in script

    def test_cluster_cap_respects_generous_declared_cpus(self):
        # 16G on 8 cores is 2048M/core, already under the cap: no bump.
        cluster = SlurmCluster(partition="compute", mem_per_cpu="3968M")
        script = self._step(
            cluster, SlurmResources(mem="16G", cpus_per_task=8)
        )
        assert "#SBATCH --cpus-per-task=8" in script
        assert "#SBATCH --mem-per-cpu=2048M" in script

    def test_cluster_cap_converts_orchestrator_mem(self):
        # The library-default orchestrator resources declare only
        # `mem`; the cluster cap must still make them submittable.
        cluster = SlurmCluster(partition="compute", mem_per_cpu="3968M")
        res = SlurmResources(time="00:10:00", mem="1G")
        script = self._orchestrator(cluster, res)
        assert "#SBATCH --mem-per-cpu=1024M" in script
        assert "#SBATCH --mem=" not in script
        assert "#SBATCH --nodes" not in script

    def test_resource_mem_per_cpu_wins_over_cluster_cap(self):
        # An explicit per-resource mem_per_cpu is authoritative: it is
        # rendered verbatim and does not trigger the cap-based bump.
        cluster = SlurmCluster(partition="compute", mem_per_cpu="3968M")
        res = SlurmResources(mem="8G", mem_per_cpu="3968M", cpus_per_task=2)
        script = self._step(cluster, res)
        assert "#SBATCH --mem-per-cpu=3968M" in script
        assert "#SBATCH --cpus-per-task=2" in script
        assert "#SBATCH --mem=" not in script

    def test_cluster_cap_defaults_none_keeps_per_node_mem(self):
        # Clusters that accept --mem (e.g. Oscar) are unaffected: the
        # cap defaults to None, so --mem is emitted unchanged.
        cluster = SlurmCluster(partition="compute")
        assert cluster.mem_per_cpu is None
        script = self._step(cluster, SlurmResources(mem="8G"))
        assert "#SBATCH --mem=8G" in script
        assert "#SBATCH --mem-per-cpu" not in script
