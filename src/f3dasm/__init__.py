"""
f3dasm - A package for data-driven design and analysis of structures
 and materials

This package provides tools for designing and optimizing materials, including
functions for data analysis, design of experiments, machine learning,
 optimization, sampling, and simulation.

Usage:
  import f3dasm

Author: Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)
"""

#                                                                       Modules
# =============================================================================

from ._src import _imports as _imports
from ._src._argparser import HPC_JOBID
from ._src._imports import try_import
from ._src.experimentdata.experimentdata import ExperimentData
from ._src.experimentdata.experimentsample import ExperimentSample
from ._src.logger import DistributedFileHandler, logger
from ._src.run_optimization import (OptimizationResult, calculate_mean_std,
                                    run_multiple_realizations,
                                    run_optimization)

#                                                        Authorship and Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
#
# =============================================================================


__version__ = '1.4.4'


# Log welcome message and the version of f3dasm
logger.info(f"Imported f3dasm (version: {__version__})")

__all__ = [
    'datageneration',
    'design',
    'optimization',
    'ExperimentData',
    'ExperimentSample',
    'DistributedFileHandler',
    'logger',
    'run_on_experimentdata',
    'run_operation_on_experiments',
    'OptimizationResult',
    'run_multiple_realizations',
    'run_optimization',
    'HPC_JOBID',
    'try_import',
    'calculate_mean_std',
    '_imports',
]
