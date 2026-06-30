"""Local executor — runs pipeline steps in the current process."""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
import logging
import time
from dataclasses import dataclass
from pathlib import Path

# Local
from ..pipeline import Pipeline, Step
from ._runner import ExecutionContext, run_step
from .base import Executor

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


@dataclass
class LocalExecutor(Executor):
    """Execute a pipeline locally in the current process.

    :class:`DataGenerator` steps with ``parallel=True`` are run
    in ``"cluster"`` mode (i.e. one job at a time). Use
    ``parallel_mode`` to override this behaviour (e.g.
    ``"parallel"`` for multiprocessing).

    Parameters
    ----------
    parallel_mode : str
        Execution mode passed to
        :meth:`DataGenerator.call` for parallel steps.
        Defaults to ``"cluster"``.
    """

    parallel_mode: str = "cluster"

    def run(
        self,
        pipeline: Pipeline,
        project_job: str | None = None,
        rootdir: Path | None = None,
    ) -> str:
        """Execute the pipeline locally.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline to execute.
        project_job : str, optional
            Job identifier used as the run folder
            (``rootdir / project_job``). Defaults to a
            timestamp-based ID.
        rootdir : Path, optional
            Root directory under which the job folder is created.
            Defaults to the current working directory.

        Returns
        -------
        str
            The project job ID.
        """
        rootdir = rootdir if rootdir is not None else Path.cwd()
        resolved_job: str = project_job or str(int(time.time()))
        job_dir: Path = rootdir / resolved_job
        job_dir.mkdir(parents=True, exist_ok=True)

        # Flatten the pipeline into a linear sequence of
        # (step, iteration_index, total_iterations) tuples.
        flat_steps: list[tuple[Step, int, int]] = pipeline._flatten()

        for step, iteration, n_iterations in flat_steps:
            if n_iterations > 1:
                logger.info(
                    f"[iter {iteration + 1}/{n_iterations}] "
                    f"Running step: {step.name}"
                )
            else:
                logger.info(f"Running step: {step.name}")

            step_dir: Path = job_dir / step.project_dir
            step_dir.mkdir(parents=True, exist_ok=True)
            _run_step_locally(
                step=step,
                run_dir=step_dir,
                parallel_mode=self.parallel_mode,
            )

        return resolved_job


def _run_step_locally(
    step: Step,
    run_dir: Path,
    parallel_mode: str = "cluster",
) -> None:
    """Execute a single pipeline step in the local process.

    Builds the local :class:`ExecutionContext` and delegates to the shared
    :func:`run_step` dispatcher. A parallel :class:`DataGenerator` step uses
    ``parallel_mode`` (default ``"cluster"``); every other step uses
    ``"cluster"`` -- the mode is only consulted for DataGenerator steps.

    Parameters
    ----------
    step : Step
        The step to execute.
    run_dir : Path
        The project run directory on disk.
    parallel_mode : str
        Mode for parallel DataGenerator steps. Defaults to ``"cluster"``.
    """
    mode = parallel_mode if step.parallel else "cluster"
    run_step(step=step, run_dir=run_dir, ctx=ExecutionContext(mode=mode))
