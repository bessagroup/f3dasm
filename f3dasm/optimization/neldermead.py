from dataclasses import dataclass

from .adapters.scipy_implementations import SciPyMinimizeOptimizer
from ..base.optimization import OptimizerParameters


@dataclass
class NelderMead_Parameters(OptimizerParameters):
    """Hyperparameters for NelderMead optimizer"""

    xatol: float = 0.0
    fatol: float = 0.0
    adaptive: bool = False
    method: str = "Nelder-Mead"


class NelderMead(SciPyMinimizeOptimizer):
    """Nelder-Mead"""

    parameter: NelderMead_Parameters = NelderMead_Parameters()
