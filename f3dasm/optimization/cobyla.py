from dataclasses import dataclass

from .adapters.scipy_implementations import SciPyMinimizeOptimizer
from ..base.optimization import OptimizerParameters


@dataclass
class COBYLA_Parameters(OptimizerParameters):
    """Hyperparameters for COBYLA optimizer"""

    tol: float = 0.0
    rhobeg: float = 1.0
    catol: float = 0.0
    method: str = "COBYLA"
    maxiter: int = 1e7


class COBYLA(SciPyMinimizeOptimizer):
    """Minimize a scalar function of one or more variables using the Constrained Optimization BY Linear Approximation (COBYLA) algorithm."""

    parameter: COBYLA_Parameters = COBYLA_Parameters()
