"""
Some API information about the opitmizers
"""
#                                                                       Modules
# =============================================================================

# Locals
from .adam import Adam, Adam_Parameters
from .adamax import Adamax, Adamax_Parameters
from .bayesianoptimization import (BayesianOptimization,
                                   BayesianOptimization_Parameters)
from .cg import CG, CG_Parameters
from .cmaes import CMAES, CMAES_Parameters
from .cmaesadam import CMAESAdam
from .differentialevolution import (DifferentialEvolution,
                                    DifferentialEvolution_Parameters)
from .ftrl import Ftrl, Ftrl_Parameters
from .lbfgsb import LBFGSB, LBFGSB_Parameters
from .nadam import Nadam, Nadam_Parameters
from .neldermead import NelderMead, NelderMead_Parameters
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

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

# List of all available optimizers
OPTIMIZERS = [
    # CMAESAdam,
    Adam,
    SGD,
    CG,
    CMAES,
    DifferentialEvolution,
    SimulatedAnnealing,
    LBFGSB,
    NelderMead,
    PSO,
    RandomSearch,
    SGA,
    SEA,
    XNES,
    RMSprop,
    Nadam,
    Adamax,
    Ftrl,
    SADE,
]
