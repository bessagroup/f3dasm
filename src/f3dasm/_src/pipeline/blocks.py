"""Built-in convenience blocks for common pipeline operations."""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
import logging
import shutil
from pathlib import Path

# Local
from ..core import Block
from ..experimentdata import ExperimentData

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================
#
# =============================================================================


logger = logging.getLogger("f3dasm")


class CollectArrayResults(Block):
    """Collect results from SLURM array job JSON files.

    This block wraps the common post-processing pattern of calling
    :meth:`ExperimentData.update_from_experimentssample_json` and
    then cleaning up the ``experiment_sample`` directory.

    Parameters
    ----------
    cleanup : bool
        Whether to remove the ``experiment_sample`` directory
        after collection. Defaults to ``True``.
    """

    def __init__(self, cleanup: bool = True) -> None:
        self.cleanup = cleanup

    def call(self, data: ExperimentData, **kwargs) -> ExperimentData:
        """Collect array results and optionally clean up.

        Parameters
        ----------
        data : ExperimentData
            The experiment data to update.
        **kwargs : dict
            Unused.

        Returns
        -------
        ExperimentData
            A new ExperimentData with collected results merged in.
        """
        logger.info("Collecting array job results from JSON files")

        # update_from_experimentssample_json returns a new
        # ExperimentData by default (in_place=False), so we
        # capture and return the result.
        data = data.update_from_experimentssample_json()

        if self.cleanup and data._project_dir is not None:
            sample_dir: Path = data._project_dir / "experiment_sample"
            if sample_dir.exists():
                logger.info(f"Cleaning up {sample_dir}")
                shutil.rmtree(sample_dir)

        return data
