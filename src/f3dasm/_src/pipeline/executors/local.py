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
from ...core import Block, DataGenerator
from ...experimentdata import ExperimentData
from ..pipeline import Pipeline, Step
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
        project_dir: str | Path | None = None,
        project_job: str | None = None,
    ) -> str:
        """Execute the pipeline locally.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline to execute.
        project_dir : str | Path, optional
            Working directory. Defaults to cwd.
        project_job : str, optional
            Project job ID. Generated if not provided.

        Returns
        -------
        str
            The project job ID.
        """
        resolved_dir: Path = Path(project_dir or Path.cwd())
        resolved_job: str = project_job or str(int(time.time()))

        run_dir: Path = resolved_dir / pipeline.name / resolved_job
        run_dir.mkdir(parents=True, exist_ok=True)

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

            _run_step_locally(
                step=step,
                run_dir=run_dir,
                parallel_mode=self.parallel_mode,
            )

        return resolved_job


def _run_step_locally(
    step: Step,
    run_dir: Path,
    parallel_mode: str = "cluster",
) -> None:
    """Execute a single pipeline step in the local process.

    Dispatches to the appropriate execution strategy based on the
    block type:

    - **DataGenerator** with ``parallel=True``: uses
      ``parallel_mode`` (default ``"cluster"``).
    - **DataGenerator** without ``parallel``: uses ``"cluster"``
      mode (single job at a time, in-process).
    - **Block**: loads ExperimentData from disk, calls
      ``arm`` + ``call``, and stores the result back.
    - **callable**: invokes with ``project_dir`` and
      ``project_job``.

    Parameters
    ----------
    step : Step
        The step to execute.
    run_dir : Path
        The project run directory on disk.
    parallel_mode : str
        Mode for DataGenerator parallel steps.
    """
    block = step.block

    if isinstance(block, DataGenerator):
        # Load ExperimentData from disk and run the DataGenerator.
        data: ExperimentData = ExperimentData.from_file(project_dir=run_dir)
        mode: str = parallel_mode if step.parallel else "cluster"
        result: ExperimentData | None = block.call(
            data=data, mode=mode, **step.kwargs
        )
        if mode == "sequential":
            result.store()
    elif isinstance(block, Block):
        # Load ExperimentData from disk, run arm + call, persist.
        data = ExperimentData.from_file(project_dir=run_dir)
        block.arm(data)
        result = block.call(data=data, **step.kwargs)
        result.store()
    elif callable(block):
        # Plain callable (e.g. the first step that creates
        # ExperimentData from scratch). It receives the run
        # directory and is responsible for creating and storing
        # data on disk.
        block(project_dir=run_dir, **step.kwargs)
    else:
        raise TypeError(
            f"Step {step.name!r} has an unsupported block type: {type(block)}"
        )
