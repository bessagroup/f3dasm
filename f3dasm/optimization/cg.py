from dataclasses import dataclass

from .adapters.scipy_implementations import SciPyMinimizeOptimizer
from ..base.optimization import OptimizerParameters


@dataclass
class CG_Parameters(OptimizerParameters):
    """Hyperparameters for CG optimizer"""

    gtol: float = 0.0
    method: str = "CG"


class CG(SciPyMinimizeOptimizer):
    """CG"""

    parameter: CG_Parameters = CG_Parameters()
