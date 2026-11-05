"""Shared builder for the golden-script pipeline.

The golden files under ``tests/pipeline/golden/`` were captured from
the renderer *before* wave-based array submission was introduced (at
commit 495b01a, where ``SlurmResources`` had no ``max_jobs_per_task``
field). Rendering the same pipeline with ``max_jobs_per_task=None`` on
every parallel step must reproduce them byte-for-byte: opting out of
waves is guaranteed to leave generated scripts unchanged.
"""

from pathlib import Path

from f3dasm._src.pipeline.executors.slurm import SlurmExecutor
from f3dasm._src.pipeline.loop import Loop
from f3dasm._src.pipeline.pipeline import Pipeline, Step
from f3dasm._src.pipeline.resources import SlurmCluster, SlurmResources

GOLDEN_DIR = Path(__file__).parent / "golden"
GOLDEN_ROOTDIR = Path("/golden")
GOLDEN_PROJECT_JOB = "GOLDEN"


def _noop(**kwargs) -> None:
    pass


def build_golden_cluster() -> SlurmCluster:
    return SlurmCluster(
        partition="compute",
        account="proj123",
        env_setup=["module load python/3.11"],
        env_vars={"MY_VAR": "value"},
        runner="python",
    )


def build_golden_pipeline(
    parallel_res_extra: dict | None = None,
) -> Pipeline:
    """Build the representative pipeline behind the golden scripts.

    Parameters
    ----------
    parallel_res_extra : dict, optional
        Extra keyword arguments for the *parallel* steps'
        :class:`SlurmResources` (e.g. ``{"max_jobs_per_task": None}``).
    """
    extra = parallel_res_extra or {}
    parallel_res = SlurmResources(
        time="02:00:00",
        mem="8G",
        cpus_per_task=4,
        max_array_size=100,
        max_concurrent=32,
        **extra,
    )
    serial_res = SlurmResources()

    return Pipeline(
        name="golden",
        steps=[
            Step(block=_noop, name="make", resources=serial_res),
            Step(
                block=_noop,
                name="evaluate",
                parallel=True,
                resources=parallel_res,
                project_dir="eval",
            ),
            Loop(
                n_iterations=3,
                steps=[
                    Step(block=_noop, name="sample", resources=serial_res),
                    Step(
                        block=_noop,
                        name="run",
                        parallel=True,
                        resources=parallel_res,
                        dependency="afterany",
                    ),
                ],
            ),
            Step(block=_noop, name="collect", resources=serial_res),
        ],
    )


def render_golden_scripts(
    parallel_res_extra: dict | None = None,
) -> dict[str, str]:
    """Render all scripts of the golden pipeline."""
    executor = SlurmExecutor(cluster=build_golden_cluster())
    return executor.generate_scripts(
        pipeline=build_golden_pipeline(parallel_res_extra),
        project_job=GOLDEN_PROJECT_JOB,
        rootdir=GOLDEN_ROOTDIR,
    )
