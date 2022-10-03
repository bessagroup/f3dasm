"""
Some API information about the opitmizers
"""

from f3dasm.optimization.adamax import Adamax
from f3dasm.optimization.nadam import Nadam
from .adam import Adam, Adam_Parameters
from .bayesianoptimization import BayesianOptimization, BayesianOptimization_Parameters
from .cg import CG, CG_Parameters
from .cmaes import CMAES, CMAES_Parameters
from .differentialevolution import DifferentialEvolution, DifferentialEvolution_Parameters
from .dualannealing import DualAnnealing, DualAnnealing_Parameters
from .lbfgsb import LBFGSB, LBFGSB_Parameters
from .momentum import Momentum, Momentum_Parameters
from .neldermead import NelderMead, NelderMead_Parameters
from .pso import PSO, PSO_Parameters
from .randomsearch import RandomSearch, RandomSearch_Parameters
from .sga import SGA, SGA_Parameters
from .xnes import XNES, XNES_Parameters
from .rmsprop import RMSprop, RMSprop_Parameters
from .nadam import Nadam, Nadam_Parameters
from .adamax import Adamax, Adamax_Parameters
from .adam2 import Adam2, Adam2_Parameters
from .sgd2 import SGD2, SGD2_Parameters

OPTIMIZERS = [
    Adam,
    CG,
    CMAES,
    DifferentialEvolution,
    DualAnnealing,
    LBFGSB,
    Momentum,
    NelderMead,
    PSO,
    RandomSearch,
    SGA,
    XNES,
    RMSprop,
    Nadam,
    Adamax,
    Adam2,
    SGD2,
]
