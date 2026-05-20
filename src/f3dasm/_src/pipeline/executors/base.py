"""Base executor interface."""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

# Local
if TYPE_CHECKING:
    from ..pipeline import Pipeline


#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================
#
# =============================================================================


class Executor(ABC):
    """Abstract base class for pipeline execution backends."""

    @abstractmethod
    def run(
        self,
        pipeline: Pipeline,
        project_job: str | None = None,
        rootdir: Path | None = None,
    ) -> str:
        """Execute a pipeline and return the project job ID.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline to execute.
        project_job : str, optional
            Existing project job ID for resumption. If ``None``,
            a timestamp-based ID is generated.
        rootdir : Path, optional
            Root directory under which the job folder is created
            (``rootdir / project_job``). Defaults to the current
            working directory.

        Returns
        -------
        str
            The project job ID.
        """
        ...
