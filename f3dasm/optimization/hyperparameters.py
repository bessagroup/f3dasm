from dataclasses import dataclass
from typing import Any, Set

from ..base.optimization import OptimizerParameters


@dataclass
class SGD_Parameters(OptimizerParameters):
    """Hyperparameters for SGD optimizer"""

    learning_rate: float = 1e-2


@dataclass
class Momentum_Parameters(OptimizerParameters):
    """Hyperparameters for Momentum optimizer"""

    learning_rate: float = 1e-2
    beta: float = 0.9


@dataclass
class Adam_Parameters(OptimizerParameters):
    """Hyperparameters for Adam optimizer"""

    learning_rate: float = 1e-2
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-8


@dataclass
class RandomSearch_Parameters(OptimizerParameters):
    """Hyperparameters for RandomSearch optimizer"""

    pass


@dataclass
class CMAES_Parameters(OptimizerParameters):
    """Hyperparameters for CMAES optimizer"""

    population: int = 30
    gen: int = 1
    memory: bool = True


@dataclass
class PSO_Parameters(OptimizerParameters):
    """Hyperparameters for PSO optimizer"""

    population: int = 30
    gen: int = 1
    memory: bool = True


@dataclass
class SGA_Parameters(OptimizerParameters):
    """Hyperparameters for SGA optimizer"""

    gen: int = 1
    cr: float = 0.9
    eta_c: float = 1.0
    m: float = 0.02
    param_m: float = 1.0
    param_s: float = 2
    crossover: str = "exponential"
    mutation: str = "polynomial"
    selection: str = "tournament"
    population: int = 30


@dataclass
class XNES_Parameters(OptimizerParameters):
    """Hyperparameters for XNES optimizer"""

    population: int = 30
    gen: int = 1
    eta_mu: float = -1.0
    eta_sigma: float = -1.0
    eta_b: float = -1.0
    sigma0: float = -1.0
    ftol: float = 1e-06
    xtol: float = 1e-06
    memory: bool = True


@dataclass
class CG_Parameters(OptimizerParameters):
    """Hyperparameters for CG optimizer"""

    gtol: float = 0.0
    method: str = "CG"


@dataclass
class NelderMead_Parameters(OptimizerParameters):
    """Hyperparameters for NelderMead optimizer"""

    xatol: float = 0.0
    fatol: float = 0.0
    adaptive: bool = False
    method: str = "Nelder-Mead"


@dataclass
class LBFGSB_Parameters(OptimizerParameters):
    """Hyperparameters for LBFGSB optimizer"""

    ftol: float = 0.0
    gtol: float = 0.0
    method: str = "L-BFGS-B"


@dataclass
class DifferentialEvolution_Parameters(OptimizerParameters):
    """Hyperparameters for DifferentialEvolution optimizer"""

    strategy: str = "best1bin"
    population: int = 15
    tol: float = 0.0
    mutation: Set = (0.5, 1)
    recombination: float = 0.7
    polish: bool = False
    atol: float = 0.0
    updating: str = "immediate"


@dataclass
class DualAnnealing_Parameters(OptimizerParameters):
    """Hyperparameters for DualAnnealing optimizer"""

    initial_temp: float = 5230.0
    restart_temp_ratio: float = 2e-05
    visit: float = 2.62
    accept: float = -5.0
    no_local_search: bool = False


@dataclass
class BayesianOptimization_Parameters(OptimizerParameters):
    """Hyperparameters for BayesianOptimization optimizer"""

    model: Any = None
    space: Any = None
    acquisition: Any = None
    evaluator: Any = None
    de_duplication: Any = None
