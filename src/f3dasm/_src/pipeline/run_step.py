"""CLI entry point invoked by SLURM jobs to run a single step.

Usage::

    python -m f3dasm._src.pipeline.run_step \\
        --step=run \\
        --job-dir=/scratch/user/1711449600 \\
        --project-dir=. \\
        --iteration=0 \\
        [--job-number=42]

This module is the *only* Python script that SLURM jobs execute.
The ``SlurmExecutor`` serialises the full :class:`Pipeline` to
``<job_dir>/.pipeline.pkl`` during submission; this entry point
deserialises it, looks up the requested step, and dispatches
execution. ExperimentData for the step lives at
``<job_dir>/<project_dir>``.
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
import argparse
import logging
import sys
from pathlib import Path

# Third-party
import cloudpickle

# Local
from ..core import Block, DataGenerator
from ..experimentdata import ExperimentData
from .loop import Loop
from .pipeline import Pipeline, Step

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================
#
# =============================================================================

logger = logging.getLogger("f3dasm")

# =============================================================================


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and execute a single pipeline step."""
    parser = argparse.ArgumentParser(description="Run a single pipeline step.")
    parser.add_argument(
        "--step",
        required=True,
        type=str,
        help="Name of the step to run.",
    )
    parser.add_argument(
        "--job-dir",
        required=True,
        type=str,
        help="Absolute path to the job directory (rootdir/project_job).",
    )
    parser.add_argument(
        "--project-dir",
        required=False,
        type=str,
        default=".",
        help="Relative sub-path within job_dir where ExperimentData lives.",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=0,
        help="Loop iteration index.",
    )
    parser.add_argument(
        "--job-number",
        type=int,
        default=None,
        help="SLURM array task ID for parallel steps.",
    )

    args = parser.parse_args(argv)

    job_dir: Path = Path(args.job_dir)
    run_dir: Path = job_dir / args.project_dir

    # --- Load the serialized pipeline definition ---
    pipeline_path: Path = job_dir / ".pipeline.pkl"
    if not pipeline_path.exists():
        logger.error(f"Pipeline file not found: {pipeline_path}")
        sys.exit(1)

    with open(pipeline_path, "rb") as f:
        pipeline: Pipeline = cloudpickle.load(f)  # noqa: S301

    # --- Find the requested step by name ---
    step: Step | None = _find_step(pipeline, args.step)
    if step is None:
        logger.error(f"Step {args.step!r} not found in pipeline.")
        sys.exit(1)

    # --- Dispatch execution ---
    run_dir.mkdir(parents=True, exist_ok=True)
    _execute_step(
        step=step,
        run_dir=run_dir,
        job_number=args.job_number,
    )


def _find_step(pipeline: Pipeline, step_name: str) -> Step | None:
    """Find a step by name in a pipeline.

    Searches both top-level steps and steps inside loops.

    Parameters
    ----------
    pipeline : Pipeline
        The pipeline to search.
    step_name : str
        The name of the step to find.

    Returns
    -------
    Step | None
        The matching step, or ``None`` if not found.
    """
    for element in pipeline.steps:
        if isinstance(element, Step) and element.name == step_name:
            return element
        elif isinstance(element, Loop):
            for s in element.steps:
                if s.name == step_name:
                    return s
    return None


def _execute_step(
    step: Step,
    run_dir: Path,
    job_number: int | None,
) -> None:
    """Execute a single step's block on a cluster node.

    This function is called from within a SLURM job. The
    dispatch logic differs from local execution:

    - **Parallel DataGenerator** (``job_number`` is set): uses
      ``"cluster_array"`` mode so each array task processes one
      job index.
    - **Non-parallel DataGenerator**: uses ``"cluster"`` mode
      with file-lock coordination for multi-node execution.
    - **Block**: loads ExperimentData, runs ``arm`` + ``call``,
      and stores the result back to disk.
    - **callable**: invokes with project context.

    Parameters
    ----------
    step : Step
        The pipeline step to execute.
    run_dir : Path
        Path to the step's ExperimentData directory on disk
        (``job_dir / step.project_dir``).
    job_number : int | None
        SLURM array task ID, or ``None`` for non-array jobs.
    """
    block = step.block

    if isinstance(block, DataGenerator):
        # Load ExperimentData from disk before dispatching.
        data: ExperimentData = ExperimentData.from_file(project_dir=run_dir)
        if step.parallel and job_number is not None:
            # Array job: each SLURM task processes one job index.
            # The DataGenerator handles the strided access pattern
            # internally (job_number::max_array_size).
            block.call(
                data=data,
                mode="cluster_array",
                job_number=job_number,
                **step.kwargs,
            )
        else:
            # Non-parallel DataGenerator on cluster: use file-lock
            # coordination so multiple nodes can safely share the
            # same ExperimentData on disk.
            block.call(data=data, mode="cluster", **step.kwargs)
    elif isinstance(block, Block):
        # Single-node Block execution: load data, transform,
        # persist.
        data: ExperimentData = ExperimentData.from_file(project_dir=run_dir)
        block.arm(data)
        result: ExperimentData = block.call(data=data, **step.kwargs)
        result.store(run_dir)
    elif callable(block):
        block(project_dir=run_dir, **step.kwargs)
    else:
        raise TypeError(
            f"Step {step.name!r} has an unsupported block type: {type(block)}"
        )


if __name__ == "__main__":
    main()
