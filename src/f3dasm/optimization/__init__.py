"""
Some API information about the opitmizers
"""
#                                                                       Modules
# =============================================================================

# Standard
import sys
from itertools import chain
from os import path
from typing import TYPE_CHECKING

# Local
from .._imports import _IntegrationModule

if TYPE_CHECKING:
    from .adam import Adam, Adam_Parameters
    from .adamax import Adamax, Adamax_Parameters
    from .bayesianoptimization import (BayesianOptimization,
                                       BayesianOptimization_Parameters)
    from .cg import CG, CG_Parameters
    from .cmaes import CMAES, CMAES_Parameters
    from .differentialevolution import (DifferentialEvolution,
                                        DifferentialEvolution_Parameters)
    from .ftrl import Ftrl, Ftrl_Parameters
    from .lbfgsb import LBFGSB, LBFGSB_Parameters
    from .nadam import Nadam, Nadam_Parameters
    from .neldermead import NelderMead, NelderMead_Parameters
    from .optimizer import Optimizer
    from .pso import PSO, PSO_Parameters
    from .randomsearch import RandomSearch, RandomSearch_Parameters
    from .rmsprop import RMSprop, RMSprop_Parameters
    from .sade import SADE, SADE_Parameters
    from .sea import SEA, SEA_Parameters
    from .sga import SGA, SGA_Parameters
    from .sgd import SGD, SGD_Parameters
    from .simulatedannealing import (SimulatedAnnealing,
                                     SimulatedAnnealing_Parameters)
    from .xnes import XNES, XNES_Parameters


# # Pygmo implementations
# try:
#     from .cmaes import CMAES, CMAES_Parameters
#     from .differentialevolution import (DifferentialEvolution,
#                                         DifferentialEvolution_Parameters)
#     from .pso import PSO, PSO_Parameters
#     from .sade import SADE, SADE_Parameters
#     from .sea import SEA, SEA_Parameters
#     from .sga import SGA, SGA_Parameters
#     from .simulatedannealing import (SimulatedAnnealing,
#                                      SimulatedAnnealing_Parameters)
#     from .xnes import XNES, XNES_Parameters
#     has_pygmo = True
# except ImportError:
#     has_pygmo = False  # skip these optimizers if pygmo is not installed

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

_import_structure: dict = {
    "optimizer": ["Optimizer"],
    "utils": ["find_optimizer", "create_optimizer_from_json", "create_optimizer_from_dict"],
    "all_optimizers": ["OPTIMIZERS"],
    "adam": ["Adam", "Adam_Parameters"],
    "adamax": ["Adamax", "Adamax_Parameters"],
    "bayesianoptimization": ["BayesianOptimization", "BayesianOptimization_Parameters"],
    "cg": ["CG", "CG_Parameters"],
    "cmaes": ["CMAES", "CMAES_Parameters"],
    "differentialevolution": ["DifferentialEvolution", "DifferentialEvolution_Parameters"],
    "ftrl": ["Ftrl", "Ftrl_Parameters"],
    "lbfgsb": ["LBFGSB", "LBFGSB_Parameters"],
    "nadam": ["Nadam", "Nadam_Parameters"],
    "neldermead": ["NelderMead", "NelderMead_Parameters"],
    "pso": ["PSO", "PSO_Parameters"],
    "randomsearch": ["RandomSearch", "RandomSearch_Parameters"],
    "rmsprop": ["RMSprop", "RMSprop_Parameters"],
    "sade": ["SADE", "SADE_Parameters"],
    "sea": ["SEA", "SEA_Parameters"],
    "sga": ["SGA", "SGA_Parameters"],
    "sgd": ["SGD", "SGD_Parameters"],
    "simulatedannealing": ["SimulatedAnnealing", "SimulatedAnnealing_Parameters"],
    "xnes": ["XNES", "XNES_Parameters"]
}

if not TYPE_CHECKING:
    class _LocalIntegrationModule(_IntegrationModule):
        __file__ = globals()["__file__"]
        __path__ = [path.dirname(__file__)]
        __all__ = list(chain.from_iterable(_import_structure.values()))
        _import_structure = _import_structure

    sys.modules[__name__] = _LocalIntegrationModule(__name__)


# pygmo_optimizers = []

# # List of all available optimizers
# OPTIMIZERS = [
#     Adam,
#     SGD,
#     CG,
#     LBFGSB,
#     NelderMead,
#     RandomSearch,
#     RMSprop,
#     Nadam,
#     Adamax,
#     Ftrl,

# ]

# # Check if pygmo has been loaded
# if has_pygmo:
#     OPTIMIZERS.extend([
#         CMAES,
#         PSO,
#         SGA,
#         SEA,
#         XNES,
#         SADE,
#         DifferentialEvolution,
#         SimulatedAnnealing,
#     ])


# def find_optimizer(query: str) -> Optimizer:
#     """Find a optimizer from the f3dasm.optimizer submodule

#     Parameters
#     ----------
#     query
#         string representation of the requested optimizer

#     Returns
#     -------
#         class of the requested optimizer
#     """
#     try:
#         return list(filter(lambda optimizer: optimizer.__name__ == query, OPTIMIZERS))[0]
#     except IndexError:
#         return ValueError(f'Optimizer {query} not found!')


# def create_optimizer_from_json(json_string: str):
#     """Create an Optimizer object from a json string

#     Parameters
#     ----------
#     json_string
#         json string representation of the information to construct the Optimizer

#     Returns
#     -------
#         Requested Optimizer object
#     """
#     optimizer_dict, name = json.loads(json_string)
#     return create_optimizer_from_dict(optimizer_dict, name)


# def create_optimizer_from_dict(optimizer_dict: dict, name: str) -> Optimizer:
#     """Create an Optimizer object from a dictionary

#     Parameters
#     ----------
#     optimizer_dict
#         dictionary representation of the information to construct the Optimizer
#     name
#         name of the class

#     Returns
#     -------
#         Requested Optimizer object
#     """
#     optimizer_dict['data'] = create_experimentdata_from_json(optimizer_dict['data'])
#     return find_optimizer(name)(**optimizer_dict)
