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
from .ftrl import Ftrl, Ftrl_Parameters
from .lbfgsb import LBFGSB, LBFGSB_Parameters
from .nadam import Nadam, Nadam_Parameters
from .neldermead import NelderMead, NelderMead_Parameters
from .randomsearch import RandomSearch, RandomSearch_Parameters
from .rmsprop import RMSprop, RMSprop_Parameters
from .sgd import SGD, SGD_Parameters

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
