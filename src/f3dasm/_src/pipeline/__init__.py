"""Pipeline module for composing and executing f3dasm blocks."""

#                                                                       Modules
# =============================================================================

# Local
from .blocks import CollectArrayResults
from .loop import Loop
from .pipeline import Pipeline, Step
from .resources import SlurmCluster, SlurmResources

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================
#
# =============================================================================


__all__ = [
    "CollectArrayResults",
    "Loop",
    "Pipeline",
    "Step",
    "SlurmCluster",
    "SlurmResources",
]
