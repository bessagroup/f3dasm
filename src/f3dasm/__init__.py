"""
F3DASM
======

This is the initalizer of the F3DASM package
"""

#                                                                       Modules
# =============================================================================

# Submodules
from f3dasm import _logging
from f3dasm import (data, experiment, functions, machinelearning,
                    optimization, sampling, simulation)

# Other utility functions
from ._show_versions import __version__, show_versions
# Design classes are accessible from the root
from .design.design import DesignSpace, make_nd_continuous_design
from .design.experimentdata import ExperimentData
from .design.parameter import (CategoricalParameter, ConstantParameter,
                               ConstraintInterface, ContinuousParameter,
                               DiscreteParameter)
# Base classes that are accessible from the root
from .functions.function import Function
from .machinelearning.model import Model
from .optimization.optimizer import Optimizer
from .run_optimization import (OptimizationResult,
                               create_optimizationresult_from_json,
                               run_multiple_realizations, run_optimization, 
                               run_multi_fidelity_optimization)
from .sampling.sampler import Sampler
from .utils import find_class, write_json

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'


# =============================================================================
#
# =============================================================================
