"""Public ``python -m`` entry point for counting open experiments.

The SLURM orchestrator calls this module to determine the SLURM
array width for a parallel step at submission time. The
implementation lives in :mod:`f3dasm._src.pipeline.count_open`.
"""

#                                                                       Modules
# =============================================================================

from .._src.pipeline.count_open import main

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
