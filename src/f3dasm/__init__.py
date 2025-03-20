"""
f3dasm - A package for data-driven design and analysis of structures
and materials

This package provides tools for designing and optimizing materials, including
functions for data analysis, design of experiments, machine learning,
optimization, sampling, and simulation.

- Documentation: https://f3dasm.readthedocs.io
- Author: Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)
"""

#                                                                       Modules
# =============================================================================

from .__version__ import __version__
from ._src._argparser import HPC_JOBID
from ._src.core import Block
from ._src.datageneration.datagenerator_factory import (create_datagenerator,
                                                        datagenerator)
from ._src.experimentdata import ExperimentData
from ._src.experimentsample import ExperimentSample
from ._src.logger import DistributedFileHandler, logger
from ._src.optimization.optimizer_factory import create_optimizer
from ._src.samplers import create_sampler

#                                                        Authorship and Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
#
# =============================================================================


# Log welcome message and the version of f3dasm
logger.info(f"Imported f3dasm (version: {__version__})")

__all__ = [
    'ExperimentData',
    'ExperimentSample',
    'create_datagenerator',
    'datagenerator',
    'create_optimizer',
    'create_sampler',
    'Block',
    'DistributedFileHandler',
    'logger',
    'HPC_JOBID',
]
