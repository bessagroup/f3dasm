"""AgenticOptimizer — f3dasm Optimizer backed by an agentic run."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .agent_runtime import AgenticRun

__author__ = "Elvis Aguero (elvis_alexander_aguero_vera@brown.edu)"
__credits__ = ["Elvis Aguero"]
__status__ = "Experimental"

__all__ = ["AgenticOptimizer"]


class AgenticOptimizer:
    """f3dasm Optimizer backed by an :class:`~agent_runtime.AgenticRun`.

    Implements the standard ``forward(ExperimentData) -> ExperimentData``
    interface so it can be used anywhere a regular f3dasm Optimizer is
    used — including as a subject of ADAS-style meta-search.

    Parameters
    ----------
    study_dir : Path
        Root of the study tree.  Must contain ``PROBLEM_STATEMENT.md``.
    **kwargs
        Forwarded to :class:`~agent_runtime.AgenticRun`.
    """

    def __init__(self, study_dir: Path, **kwargs: Any) -> None:
        self._run = AgenticRun(study_dir, **kwargs)

    def forward(self, data: Any) -> Any:
        """Run the agentic loop and return data with updated outputs.

        Serialises open jobs in *data* into the study workspace, calls
        :meth:`~agent_runtime.AgenticRun.execute`, and reads results back
        into *data*.

        Parameters
        ----------
        data : ExperimentData
            f3dasm experiment data containing open (pending) evaluation jobs.

        Returns
        -------
        ExperimentData
            The same object with agent-produced outputs filled in.

        Raises
        ------
        NotImplementedError
            The serialisation contract (how open jobs are passed to agents
            and how outputs are read back) is not yet implemented.
        """
        raise NotImplementedError(
            "AgenticOptimizer.forward() is not yet implemented. "
            "The serialisation contract between ExperimentData and the "
            "agentic workspace is pending. Use AgenticRun.execute() directly "
            "for standalone agentic runs."
        )
