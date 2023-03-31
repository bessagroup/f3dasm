"""
F3DASM - A package for data-driven design and analysis of structures and materials

This package provides tools for designing and optimizing materials, including
functions for data analysis, design of experiments, machine learning, optimization,
sampling, and simulation.

Usage:
  import f3dasm

Author: Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)
"""

#                                                                       Modules
# =============================================================================

from f3dasm import (_logging, data, design, experiment, functions,
                    machinelearning, optimization, sampling, simulation)

from ._show_versions import __version__, show_versions
# Design classes
from .design.design import DesignSpace, make_nd_continuous_design
from .design.experimentdata import ExperimentData
from .design.parameter import (CategoricalParameter, ConstantParameter,
                               ConstraintInterface, ContinuousParameter,
                               DiscreteParameter)
# Base classes
from .functions.function import Function
from .machinelearning.model import Model
from .optimization.optimizer import Optimizer
from .run_optimization import (OptimizationResult,
                               create_optimizationresult_from_json,
                               run_multiple_realizations, run_optimization)
from .sampling.sampler import Sampler
from .utils import find_class, write_json

#                                                        Authorship and Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
