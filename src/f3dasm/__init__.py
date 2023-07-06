"""
f3dasm - A package for data-driven design and analysis of structures and materials

This package provides tools for designing and optimizing materials, including
functions for data analysis, design of experiments, machine learning, optimization,
sampling, and simulation.

Usage:
  import f3dasm

Author: Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)
"""

#                                                                       Modules
# =============================================================================

from . import datageneration, design, machinelearning, optimization, sampling
from .argparser import HPC_JOBID
from .datageneration.functions.function import Function
from .design.design import Design
from .design.domain import Domain, make_nd_continuous_domain
from .design.experimentdata import ExperimentData
from .design.parameter import (CategoricalParameter, ConstantParameter,
                               ContinuousParameter, DiscreteParameter)
from .logger import DistributedFileHandler, logger
from .optimization.optimizer import Optimizer
from .run_optimization import (OptimizationResult, run_multiple_realizations,
                               run_optimization)
from .sampling.sampler import Sampler

#                                                        Authorship and Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
#
# =============================================================================

__version__ = '1.2.0'

# Log welcome message and the version of f3dasm
logger.info(f"Imported f3dasm (version: {__version__})")

__all__ = [
    'datageneration',
    'design',
    'machinelearning',
    'optimization',
    'sampling',
    'Function',
    'Domain',
    'make_nd_continuous_domain',
    'ExperimentData',
    'CategoricalParameter',
    'ConstantParameter',
    'ContinuousParameter',
    'DiscreteParameter',
    'DistributedFileHandler',
    'logger',
    'Optimizer',
    'run_on_experimentdata',
    'run_operation_on_experiments',
    'OptimizationResult',
    'run_multiple_realizations',
    'run_optimization',
    'Sampler'
]
