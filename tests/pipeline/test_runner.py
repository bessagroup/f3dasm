"""Tests for the unified step runner (run_step + ExecutionContext)."""

import pytest

from f3dasm import ExperimentData, create_sampler, datagenerator
from f3dasm._src.pipeline.executors._runner import (
    ExecutionContext,
    run_step,
)
from f3dasm._src.pipeline.pipeline import Step
from f3dasm.design import Domain

pytestmark = pytest.mark.smoke


@datagenerator(output_names="y")
def _parallel_const(x):
    """Module-level generator so multiprocessing can pickle it."""
    return 7.0


def _stored_open_data(tmp_path, n_samples=3):
    """Store an ExperimentData with `n_samples` open jobs to disk."""
    domain = Domain()
    domain.add_float("x", 0.0, 1.0)
    domain.add_output("y")
    data = ExperimentData(domain=domain)
    sampler = create_sampler("random", seed=42)
    data = sampler.call(data=data, n_samples=n_samples)
    data.store(project_dir=tmp_path)
    return data


class TestRunStepSequential:
    def test_sequential_datagenerator_persists_outputs(self, tmp_path):
        @datagenerator(output_names="y")
        def const(x):
            return 42.0

        _stored_open_data(tmp_path, n_samples=3)
        step = Step(block=const, name="gen")

        run_step(
            step=step,
            run_dir=tmp_path,
            ctx=ExecutionContext(mode="sequential"),
        )

        reloaded = ExperimentData.from_file(project_dir=tmp_path)
        ys = [
            reloaded.get_experiment_sample(i).output_data["y"]
            for i in range(3)
        ]
        assert ys == [42.0, 42.0, 42.0]


class TestRunStepBlock:
    def test_block_result_is_persisted(self, tmp_path):
        from f3dasm._src.core import Block

        class MarkAllFinished(Block):
            def call(self, data, **kwargs):
                return data.mark_all("finished")

        _stored_open_data(tmp_path, n_samples=3)
        step = Step(block=MarkAllFinished(), name="mark")

        run_step(
            step=step,
            run_dir=tmp_path,
            ctx=ExecutionContext(mode="sequential"),
        )

        reloaded = ExperimentData.from_file(project_dir=tmp_path)
        assert len(reloaded.select_with_status("open")) == 0


class TestRunStepCallable:
    def test_callable_invoked_with_project_dir_and_kwargs(self, tmp_path):
        received = {}

        def make(project_dir, **kwargs):
            received["project_dir"] = str(project_dir)
            received["kwargs"] = dict(kwargs)

        step = Step(block=make, name="create", kwargs={"n": 5})

        run_step(
            step=step,
            run_dir=tmp_path,
            ctx=ExecutionContext(mode="sequential"),
        )

        assert received["project_dir"] == str(tmp_path)
        assert received["kwargs"] == {"n": 5}

    def test_unsupported_block_type_raises(self, tmp_path):
        step = Step(block=42, name="bad")
        with pytest.raises(TypeError, match="unsupported block type"):
            run_step(
                step=step,
                run_dir=tmp_path,
                ctx=ExecutionContext(mode="sequential"),
            )


class TestRunStepArrayStriding:
    def test_array_task_processes_strided_subset(self, tmp_path):
        @datagenerator(output_names="y")
        def const(x):
            return 1.0

        _stored_open_data(tmp_path, n_samples=4)
        step = Step(block=const, name="gen", parallel=True)

        # Array task 0 of width 2 owns the strided open indices [0, 2].
        run_step(
            step=step,
            run_dir=tmp_path,
            ctx=ExecutionContext(
                mode="cluster_array", job_number=0, max_array_size=2
            ),
        )

        sample_dir = tmp_path / "experiment_sample"
        written = sorted(p.stem for p in sample_dir.glob("*.json"))
        assert written == ["0", "2"]


class TestRunStepWaves:
    def test_wave_task_processes_single_offset_job(self, tmp_path):
        @datagenerator(output_names="y")
        def const(x):
            return 1.0

        _stored_open_data(tmp_path, n_samples=5)
        step = Step(block=const, name="gen", parallel=True)

        # Wave 1 of width 2 with one job per task: task 0 owns
        # open[1*2*1 + 0] = index 2 and nothing else.
        run_step(
            step=step,
            run_dir=tmp_path,
            ctx=ExecutionContext(
                mode="cluster_array",
                job_number=0,
                max_array_size=2,
                wave=1,
                max_jobs_per_task=1,
            ),
        )

        sample_dir = tmp_path / "experiment_sample"
        written = sorted(p.stem for p in sample_dir.glob("*.json"))
        assert written == ["2"]

    @pytest.mark.parametrize(
        "n_open,max_array_size,max_jobs_per_task",
        [
            (9, 2, 1),
            (10, 2, 1),
            (9, 3, 2),
            (13, 4, 3),
            (3, 900, 1),
            (7, 3, None),
        ],
    )
    def test_waves_partition_open_jobs(
        self, n_open, max_array_size, max_jobs_per_task
    ):
        """Waves x tasks partition the open jobs: disjoint, complete,
        and each task owns at most ``max_jobs_per_task`` jobs."""
        open_jobs = list(range(n_open))
        w, j = max_array_size, max_jobs_per_task
        if j is None:
            n_waves = 1
        else:
            n_waves = -(-n_open // (w * j))  # ceil

        seen = []
        for wave in range(n_waves):
            for task in range(w):
                ctx = ExecutionContext(
                    mode="cluster_array",
                    job_number=task,
                    max_array_size=w,
                    wave=wave,
                    max_jobs_per_task=j,
                )
                assigned = ctx.assigned_jobs(open_jobs)
                if j is not None:
                    assert len(assigned) <= j
                seen.extend(assigned)

        assert sorted(seen) == open_jobs
        assert len(seen) == len(set(seen))


class TestParallelModePersistRegression:
    """parallel_mode='parallel' must persist its results (was a latent gap)."""

    def test_parallel_mode_persists_results(self, tmp_path):
        from f3dasm._src.pipeline.executors.local import _run_step_locally

        _stored_open_data(tmp_path, n_samples=2)
        step = Step(block=_parallel_const, name="gen", parallel=True)

        _run_step_locally(
            step=step, run_dir=tmp_path, parallel_mode="parallel"
        )

        reloaded = ExperimentData.from_file(project_dir=tmp_path)
        ys = [
            reloaded.get_experiment_sample(i).output_data["y"]
            for i in range(2)
        ]
        assert ys == [7.0, 7.0]
