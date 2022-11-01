from dataclasses import dataclass

from .adapters.scipy_implementations import SciPyMinimizeOptimizer
from ..base.optimization import OptimizerParameters


@dataclass
class NelderMead_Parameters(OptimizerParameters):
    """Hyperparameters for NelderMead optimizer

    Args:
        xatol (float): _description_ (Default = 0.0)
        fatol (float): _description_ (Default = 0.0)
        adaptive (bool): _description_ (Default = False)
    """

    xatol: float = 0.0
    fatol: float = 0.0
    adaptive: bool = False


class NelderMead(SciPyMinimizeOptimizer):
    """Nelder-Mead"""

    method: str = "Nelder-Mead"
    parameter: NelderMead_Parameters = NelderMead_Parameters()
