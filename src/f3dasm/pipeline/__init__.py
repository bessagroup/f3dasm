"""
Module for composing and executing f3dasm pipelines.

This package is the public entry point for the pipeline system.
It re-exports the user-facing types and provides two
``python -m``-able submodules used by the SLURM executor:

- :mod:`f3dasm.pipeline.run_step` — runs a single step on a
  compute node.
- :mod:`f3dasm.pipeline.count_open` — prints the number of open
  experiments in an :class:`f3dasm.ExperimentData` on disk.
"""

#                                                                       Modules
# =============================================================================

from .._src.pipeline import (
    CollectArrayResults,
    Loop,
    Pipeline,
    SlurmCluster,
    SlurmResources,
    Step,
)

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
    "SlurmCluster",
    "SlurmResources",
    "Step",
]
