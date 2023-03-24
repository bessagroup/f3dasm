"""
Some API information about the opitmizers
"""
#                                                                       Modules
# =============================================================================

import json

from ..design import create_experimentdata_from_json
from .adam import Adam, Adam_Parameters
from .adamax import Adamax, Adamax_Parameters
from .bayesianoptimization import (BayesianOptimization,
                                   BayesianOptimization_Parameters)
from .cg import CG, CG_Parameters
from .ftrl import Ftrl, Ftrl_Parameters
from .lbfgsb import LBFGSB, LBFGSB_Parameters
from .nadam import Nadam, Nadam_Parameters
from .neldermead import NelderMead, NelderMead_Parameters
from .optimizer import Optimizer
from .randomsearch import RandomSearch, RandomSearch_Parameters
from .rmsprop import RMSprop, RMSprop_Parameters
from .sgd import SGD, SGD_Parameters

# Locals


# Pygmo implementations
try:
    from .cmaes import CMAES, CMAES_Parameters
    from .differentialevolution import (DifferentialEvolution,
                                        DifferentialEvolution_Parameters)
    from .pso import PSO, PSO_Parameters
    from .sade import SADE, SADE_Parameters
    from .sea import SEA, SEA_Parameters
    from .sga import SGA, SGA_Parameters
    from .simulatedannealing import (SimulatedAnnealing,
                                     SimulatedAnnealing_Parameters)
    from .xnes import XNES, XNES_Parameters
    has_pygmo = True
except ImportError:
    has_pygmo = False  # skip these optimizers if pygmo is not installed
from .bayesianoptimization import BayesianOptimization, BayesianOptimization_Parameters
from .bayesianoptimization_torch import BayesianOptimizationTorch, BayesianOptimizationTorch_Parameters, MFBayesianOptimizationTorch

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

pygmo_optimizers = []

# List of all available optimizers
OPTIMIZERS = [
    Adam,
    SGD,
    CG,
    LBFGSB,
    NelderMead,
    RandomSearch,
    RMSprop,
    Nadam,
    Adamax,
    Ftrl,

]

# Check if pygmo has been loaded
if has_pygmo:
    OPTIMIZERS.extend([
        CMAES,
        PSO,
        SGA,
        SEA,
        XNES,
        SADE,
        DifferentialEvolution,
        SimulatedAnnealing,
    ])


def find_optimizer(query: str) -> Optimizer:
    """Find a optimizer from the f3dasm.optimizer submodule

    Parameters
    ----------
    query
        string representation of the requested optimizer

    Returns
    -------
        class of the requested optimizer
    """
    try:
        return list(filter(lambda optimizer: optimizer.__name__ == query, OPTIMIZERS))[0]
    except IndexError:
        return ValueError(f'Optimizer {query} not found!')


def create_optimizer_from_json(json_string: str):
    """Create an Optimizer object from a json string

    Parameters
    ----------
    json_string
        json string representation of the information to construct the Optimizer

    Returns
    -------
        Requested Optimizer object
    """
    optimizer_dict, name = json.loads(json_string)
    return create_optimizer_from_dict(optimizer_dict, name)


def create_optimizer_from_dict(optimizer_dict: dict, name: str) -> Optimizer:
    """Create an Optimizer object from a dictionary

    Parameters
    ----------
    optimizer_dict
        dictionary representation of the information to construct the Optimizer
    name
        name of the class

    Returns
    -------
        Requested Optimizer object
    """
    optimizer_dict['data'] = create_experimentdata_from_json(optimizer_dict['data'])
    return find_optimizer(name)(**optimizer_dict)
