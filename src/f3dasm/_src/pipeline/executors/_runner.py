"""Unified single-step dispatch shared by every executor.

``run_step`` is the one place that knows how to run a pipeline
:class:`~f3dasm._src.pipeline.pipeline.Step`: the ``from_file -> arm``
preamble, the block-type ladder, and the single uniform persistence rule.

The per-environment difference -- running a step locally versus running it as
one task of a SLURM array -- is carried by an :class:`ExecutionContext`. That
context is a small, serializable *value* (not a behaviour object) so it can
cross the cloudpickle/CLI boundary into a fresh worker process, which never
holds the submitting executor instance.
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

# Local
from ...core import Block, DataGenerator
from ...experimentdata import ExperimentData

if TYPE_CHECKING:
    from ..pipeline import Step

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================
#
# =============================================================================


@dataclass(frozen=True)
class ExecutionContext:
    """How and where a single step is evaluated, independent of what it runs.

    Parameters
    ----------
    mode : str
        The parallelization mode passed to :meth:`DataGenerator.call` for a
        single (non-array) invocation. One of ``"sequential"``,
        ``"parallel"``, or ``"cluster"``.
    job_number : int | None
        For a SLURM array task, the array-task index this invocation owns;
        ``None`` for a single local or whole-cluster invocation.
    max_array_size : int | None
        For a SLURM array task, the array width used to stride the open jobs
        across tasks. Unused when ``job_number`` is ``None``.
    """

    mode: str
    job_number: int | None = None
    max_array_size: int | None = None

    def execute(
        self,
        block: DataGenerator,
        data: ExperimentData,
        kwargs: dict,
    ) -> ExperimentData | None:
        """Run ``block`` over ``data`` according to this context.

        Returns
        -------
        ExperimentData | None
            The in-memory ExperimentData for return-and-store modes
            (``sequential``, ``parallel``); ``None`` for the self-persisting
            cluster modes, which write their results to disk directly.
        """
        if self.job_number is None:
            return block.call(data=data, mode=self.mode, **kwargs)

        # SLURM array task: own only the strided slice of the open jobs and
        # run each as a self-persisting ``cluster_array`` evaluation.
        open_jobs = data.select_with_status("open").index.tolist()
        for idx in open_jobs[self.job_number :: self.max_array_size]:
            block.call(
                data=data,
                mode="cluster_array",
                job_number=idx,
                **kwargs,
            )
        return None


def run_step(step: Step, run_dir: Path, ctx: ExecutionContext) -> None:
    """Execute a single pipeline step against the data in ``run_dir``.

    Parameters
    ----------
    step : Step
        The pipeline step to execute (its block, kwargs, and parallel flag).
    run_dir : Path
        The project run directory on disk holding the step's ExperimentData.
    ctx : ExecutionContext
        How and where to evaluate the step in the current environment.

    Raises
    ------
    TypeError
        If the step's block is neither a DataGenerator, a Block, nor callable.
    """
    block = step.block

    if isinstance(block, DataGenerator):
        data: ExperimentData = ExperimentData.from_file(project_dir=run_dir)
        block.arm(data)
        result = ctx.execute(block, data, step.kwargs)
        if result is not None:
            result.store(run_dir)
    elif isinstance(block, Block):
        data = ExperimentData.from_file(project_dir=run_dir)
        block.arm(data)
        result = block.call(data=data, **step.kwargs)
        result.store(run_dir)
    elif callable(block):
        block(project_dir=run_dir, **step.kwargs)
    else:
        raise TypeError(
            f"Step {step.name!r} has an unsupported block type: {type(block)}"
        )
