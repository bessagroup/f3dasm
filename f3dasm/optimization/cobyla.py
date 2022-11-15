from dataclasses import dataclass

from ..base.optimization import OptimizerParameters
from .adapters.scipy_implementations import SciPyMinimizeOptimizer


@dataclass
class COBYLA_Parameters(OptimizerParameters):
    """Hyperparameters for COBYLA optimizer"""

    tol: float = 0.0
    rhobeg: float = 1.0
    catol: float = 0.0
    maxiter: int = 1e7


class COBYLA(SciPyMinimizeOptimizer):
    """Minimize a scalar function of one or more variables using the Constrained Optimization BY Linear Approximation (COBYLA) algorithm."""

    method: str = "COBYLA"
    parameter: COBYLA_Parameters = COBYLA_Parameters()
