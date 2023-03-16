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
    from .all_optimizers import OPTIMIZERS
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
    from .utils import (create_optimizer_from_dict, create_optimizer_from_json,
                        find_optimizer)
    from .xnes import XNES, XNES_Parameters

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

_import_structure: dict = {
    "utils": ["create_optimizer_from_json", "create_optimizer_from_dict", "find_optimizer"],
    "optimizer": ["Optimizer"],
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
    "xnes": ["XNES", "XNES_Parameters"],
    "all_optimizers": ["OPTIMIZERS"],
}

if not TYPE_CHECKING:
    class _LocalIntegrationModule(_IntegrationModule):
        __file__ = globals()["__file__"]
        __path__ = [path.dirname(__file__)]
        __all__ = list(chain.from_iterable(_import_structure.values()))
        _import_structure = _import_structure

    sys.modules[__name__] = _LocalIntegrationModule(__name__)
