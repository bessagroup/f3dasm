"""Execution backends for f3dasm pipelines."""

#                                                                       Modules
# =============================================================================

# Local
from .base import Executor
from .local import LocalExecutor
from .slurm import SlurmExecutor

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================
#
# =============================================================================


__all__ = ["Executor", "LocalExecutor", "SlurmExecutor"]
