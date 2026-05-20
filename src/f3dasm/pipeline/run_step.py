"""Public ``python -m`` entry point for running a single pipeline step.

This module is invoked by the SLURM executor inside generated
sbatch scripts. The implementation lives in
:mod:`f3dasm._src.pipeline.run_step`; this module re-exports
``main`` and runs it under ``__main__``.
"""

#                                                                       Modules
# =============================================================================

from .._src.pipeline.run_step import main

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================
#
# =============================================================================

__all__ = ["main"]


if __name__ == "__main__":
    main()
