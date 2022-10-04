"""
Some API information about the opitmizers
"""

from .adam import Adam, Adam_Parameters
from .sgd import SGD, SGD_Parameters
from .bayesianoptimization import BayesianOptimization, BayesianOptimization_Parameters
from .cg import CG, CG_Parameters
from .cmaes import CMAES, CMAES_Parameters
from .differentialevolution import DifferentialEvolution, DifferentialEvolution_Parameters
from .dualannealing import DualAnnealing, DualAnnealing_Parameters
from .lbfgsb import LBFGSB, LBFGSB_Parameters
from .neldermead import NelderMead, NelderMead_Parameters
from .pso import PSO, PSO_Parameters
from .randomsearch import RandomSearch, RandomSearch_Parameters
from .sga import SGA, SGA_Parameters
from .xnes import XNES, XNES_Parameters
from .rmsprop import RMSprop, RMSprop_Parameters
from .nadam import Nadam, Nadam_Parameters
from .adamax import Adamax, Adamax_Parameters
from .ftrl import Ftrl, Ftrl_Parameters

OPTIMIZERS = [
    Adam,
    SGD,
    CG,
    CMAES,
    DifferentialEvolution,
    DualAnnealing,
    LBFGSB,
    NelderMead,
    PSO,
    RandomSearch,
    SGA,
    XNES,
    RMSprop,
    Nadam,
    Adamax,
    Ftrl,
]
