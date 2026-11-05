"""Execute rendered orchestrator scripts against a fake ``sbatch``.

String assertions on the rendered bash verify vocabulary, not control
flow. These tests actually *run* the orchestrator under ``bash`` with a
shim ``sbatch`` on ``PATH`` (records its argv and the exported
``F3DASM_WAVE``, echoes ``Submitted batch job <n>``) and a stub runner
that answers ``count_open`` with a fixed number, then drive the full
wake sequence: every recorded ``sbatch "$SELF" <args>`` resubmission is
fed back in as the next wake. The resulting submission trace pins down
the wave state machine — array widths per wave (including the ragged
last wave), ``afterany`` chaining between waves, the next-step
dependency attaching to the final wave, and counter resets.
"""

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from f3dasm._src.pipeline.executors.slurm import (
    render_orchestrator_script,
)
from f3dasm._src.pipeline.loop import Loop
from f3dasm._src.pipeline.pipeline import Pipeline, Step
from f3dasm._src.pipeline.resources import SlurmCluster, SlurmResources

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.skipif(
        sys.platform == "win32" or shutil.which("bash") is None,
        reason="orchestrator scripts are POSIX-only and require bash",
    ),
]

MAX_WAKES = 50


@dataclass
class SbatchCall:
    """One recorded invocation of the fake ``sbatch``."""

    job_id: str
    wave_env: str
    args: list[str]

    @property
    def dependency(self) -> str | None:
        for a in self.args:
            if a.startswith("--dependency="):
                return a.removeprefix("--dependency=")
        return None

    @property
    def array(self) -> str | None:
        for a in self.args:
            if a.startswith("--array="):
                return a.removeprefix("--array=")
        return None

    def self_args(self, orch_path: Path) -> list[str] | None:
        """The counter arguments if this call resubmits ``orch_path``."""
        target = str(orch_path.resolve())
        for i, a in enumerate(self.args):
            if a == target:
                return self.args[i + 1 :]
        return None


class OrchestratorHarness:
    """Runs an orchestrator script through its full wake sequence."""

    def __init__(self, tmp_path: Path, n_open: int):
        self.tmp_path = tmp_path
        self.shim_dir = tmp_path / "shim"
        self.shim_dir.mkdir()
        self.log_file = tmp_path / "sbatch_log"
        counter_file = tmp_path / "job_counter"
        counter_file.write_text("100")

        sbatch = self.shim_dir / "sbatch"
        sbatch.write_text(
            "#!/bin/bash\n"
            f'C="{counter_file}"\n'
            'N=$(( $(cat "$C") + 1 ))\n'
            'echo $N > "$C"\n'
            f'printf \'%s|%s|%s\\n\' "$N" "${{F3DASM_WAVE:-unset}}" "$*"'
            f' >> "{self.log_file}"\n'
            'echo "Submitted batch job $N"\n'
        )
        sbatch.chmod(0o755)

        # The orchestrator sizes arrays via
        # ``<runner> -m f3dasm.pipeline.count_open ...``; the stub
        # runner ignores its arguments and prints a fixed count.
        fakepython = self.shim_dir / "fakepython"
        fakepython.write_text(f"#!/bin/bash\necho {n_open}\n")
        fakepython.chmod(0o755)

        self.cluster = SlurmCluster(
            partition="compute", account="proj", runner="fakepython"
        )
        self.orch_path = tmp_path / "orchestrator.sh"

    def render(self, pipeline: Pipeline, script_paths: dict) -> None:
        script = render_orchestrator_script(
            pipeline=pipeline,
            cluster=self.cluster,
            orchestrator_resources=SlurmResources(),
            script_paths=script_paths,
            log_dir_path=(self.tmp_path / "logs").as_posix(),
            job_dir=self.tmp_path,
        )
        self.orch_path.write_text(script)

    def _read_calls(self) -> list[SbatchCall]:
        if not self.log_file.exists():
            return []
        calls = []
        for line in self.log_file.read_text().splitlines():
            job_id, wave_env, args = line.split("|", maxsplit=2)
            calls.append(SbatchCall(job_id, wave_env, args.split()))
        return calls

    def drive(self, initial_args=("0", "0", "0")) -> list[SbatchCall]:
        """Run wake after wake until the pipeline completes.

        Returns the full ``sbatch`` submission trace. Fails the test
        if the final wake does not print ``Pipeline complete.`` or if
        the pipeline does not converge within ``MAX_WAKES`` wakes.
        """
        env = {
            **os.environ,
            "PATH": f"{self.shim_dir}:{os.environ['PATH']}",
        }
        args = list(initial_args)
        for _ in range(MAX_WAKES):
            n_before = len(self._read_calls())
            result = subprocess.run(
                ["bash", str(self.orch_path), *args],
                env=env,
                capture_output=True,
                text=True,
                check=True,
            )
            new_calls = self._read_calls()[n_before:]
            resubmissions = [
                a
                for c in new_calls
                if (a := c.self_args(self.orch_path)) is not None
            ]
            if not resubmissions:
                assert "Pipeline complete." in result.stdout
                return self._read_calls()
            assert len(resubmissions) == 1, (
                f"one wake must resubmit SELF at most once, "
                f"got {resubmissions}"
            )
            args = resubmissions[0]
        pytest.fail(f"pipeline did not complete within {MAX_WAKES} wakes")

    def step_submissions(self, script_path: str) -> list[SbatchCall]:
        return [c for c in self._read_calls() if script_path in c.args]


def _parallel_step(max_jobs_per_task, max_array_size=3, **step_kwargs):
    res = SlurmResources(
        max_array_size=max_array_size,
        max_concurrent=16,
        max_jobs_per_task=max_jobs_per_task,
    )
    return Step(
        block=lambda: None,
        name="run",
        resources=res,
        parallel=True,
        **step_kwargs,
    )


class TestWaveStateMachine:
    def test_overflow_step_runs_three_waves(self, tmp_path):
        """N=8, W=3, j=1 -> waves of 3, 3, 2 tasks."""
        harness = OrchestratorHarness(tmp_path, n_open=8)
        pipeline = Pipeline(
            name="pipe",
            steps=[
                _parallel_step(max_jobs_per_task=1),
                Step(block=lambda: None, name="post"),
            ],
        )
        harness.render(
            pipeline,
            {"run": "SCRIPT_DIR/run.sh", "post": "SCRIPT_DIR/post.sh"},
        )
        harness.drive()

        waves = harness.step_submissions("SCRIPT_DIR/run.sh")
        assert [c.array for c in waves] == ["0-2%16", "0-2%16", "0-1%16"]
        assert [c.wave_env for c in waves] == ["0", "1", "2"]

        # Waves chain afterany on the previous wave's array job.
        resubs = [
            c
            for c in harness._read_calls()
            if c.self_args(harness.orch_path) is not None
        ]
        # Wake 1 and 2 re-enter the step for the next wave.
        assert resubs[0].dependency == f"afterany:{waves[0].job_id}"
        assert resubs[0].self_args(harness.orch_path) == ["0", "0", "0", "1"]
        assert resubs[1].dependency == f"afterany:{waves[1].job_id}"
        assert resubs[1].self_args(harness.orch_path) == ["0", "0", "0", "2"]
        # The advance past the step depends afterok on the FINAL wave.
        assert resubs[2].dependency == f"afterok:{waves[2].job_id}"
        assert resubs[2].self_args(harness.orch_path)[:2] == ["1", "0"]

    def test_exact_multiple_runs_two_full_waves(self, tmp_path):
        """N=6, W=3, j=1 -> exactly two full waves, no empty third."""
        harness = OrchestratorHarness(tmp_path, n_open=6)
        pipeline = Pipeline(name="pipe", steps=[_parallel_step(1)])
        harness.render(pipeline, {"run": "SCRIPT_DIR/run.sh"})
        harness.drive()

        waves = harness.step_submissions("SCRIPT_DIR/run.sh")
        assert [c.array for c in waves] == ["0-2%16", "0-2%16"]
        assert [c.wave_env for c in waves] == ["0", "1"]

    def test_bounded_stride_reduces_wave_count(self, tmp_path):
        """N=8, W=3, j=2 -> window 6: waves of 3 and 1 tasks."""
        harness = OrchestratorHarness(tmp_path, n_open=8)
        pipeline = Pipeline(name="pipe", steps=[_parallel_step(2)])
        harness.render(pipeline, {"run": "SCRIPT_DIR/run.sh"})
        harness.drive()

        waves = harness.step_submissions("SCRIPT_DIR/run.sh")
        # Wave 0: 3 tasks x 2 jobs; wave 1: REM=2 -> ceil(2/2)=1 task.
        assert [c.array for c in waves] == ["0-2%16", "0-0%16"]

    def test_fits_single_wave_advances_directly(self, tmp_path):
        """N=2 <= W*j: one wave, no wave resubmission."""
        harness = OrchestratorHarness(tmp_path, n_open=2)
        pipeline = Pipeline(name="pipe", steps=[_parallel_step(1)])
        harness.render(pipeline, {"run": "SCRIPT_DIR/run.sh"})
        harness.drive()

        waves = harness.step_submissions("SCRIPT_DIR/run.sh")
        assert [c.array for c in waves] == ["0-1%16"]
        resubs = [
            c
            for c in harness._read_calls()
            if c.self_args(harness.orch_path) is not None
        ]
        assert resubs[0].dependency == f"afterok:{waves[0].job_id}"

    def test_opted_out_step_submits_once(self, tmp_path):
        """j=None: a single strided array regardless of N."""
        harness = OrchestratorHarness(tmp_path, n_open=8)
        pipeline = Pipeline(name="pipe", steps=[_parallel_step(None)])
        harness.render(pipeline, {"run": "SCRIPT_DIR/run.sh"})
        harness.drive()

        waves = harness.step_submissions("SCRIPT_DIR/run.sh")
        assert [c.array for c in waves] == ["0-2%16"]
        assert [c.wave_env for c in waves] == ["unset"]

    def test_loop_inner_step_waves_reset_per_iteration(self, tmp_path):
        """A parallel inner step waves within each loop iteration."""
        harness = OrchestratorHarness(tmp_path, n_open=4)
        inner = _parallel_step(1, dependency="afterany")
        pipeline = Pipeline(
            name="pipe", steps=[Loop(n_iterations=2, steps=[inner])]
        )
        harness.render(pipeline, {"loop0_run": "SCRIPT_DIR/loop0_run.sh"})
        harness.drive()

        waves = harness.step_submissions("SCRIPT_DIR/loop0_run.sh")
        # Two iterations x (wave 0: 3 tasks, wave 1: 1 task).
        assert [c.array for c in waves] == [
            "0-2%16",
            "0-0%16",
            "0-2%16",
            "0-0%16",
        ]
        assert [c.wave_env for c in waves] == ["0", "1", "0", "1"]
