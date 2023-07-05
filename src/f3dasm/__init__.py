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

from f3dasm import (design, experiment, functions, machinelearning,
                    optimization, sampling, simulation)

from ._logging import DistributedFileHandler, logger
# Design classes
from .design.design import DesignSpace, make_nd_continuous_design
from .design.experimentdata import ExperimentData
from .design.parameter import (CategoricalParameter, ConstantParameter,
                               ContinuousParameter, DiscreteParameter)
from .experiment.parallelization import (run_on_experimentdata,
                                         run_operation_on_experiments)
# Base classes
from .functions.function import Function
from .machinelearning.model import Model
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
