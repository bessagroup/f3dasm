from dataclasses import dataclass

from ..base.optimization import OptimizerParameters
from .adapters.scipy_implementations import SciPyMinimizeOptimizer


@dataclass
class NelderMead_Parameters(OptimizerParameters):
    """Hyperparameters for NelderMead optimizer"""

    xatol: float = 0.0
    fatol: float = 0.0
    adaptive: bool = False


class NelderMead(SciPyMinimizeOptimizer):
    """Nelder-Mead"""

    method: str = "Nelder-Mead"
    parameter: NelderMead_Parameters = NelderMead_Parameters()
