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

from ._src.core import Block, DataGenerator, Optimizer, datagenerator
from ._src.datageneration.datagenerator_factory import create_datagenerator
from ._src.experimentdata import ExperimentData
from ._src.experimentsample import ExperimentSample
from ._src.optimization.optimizer_factory import create_optimizer
from ._src.samplers import create_sampler

#                                                        Authorship and Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
#
# =============================================================================

__all__ = [
    "ExperimentData",
    "ExperimentSample",
    "create_datagenerator",
    "datagenerator",
    "create_optimizer",
    "create_sampler",
    "Block",
    "Optimizer",
    "DataGenerator",
]
