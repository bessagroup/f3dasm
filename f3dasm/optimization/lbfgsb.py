from dataclasses import dataclass

from ..base.optimization import OptimizerParameters
from .adapters.scipy_implementations import SciPyMinimizeOptimizer


@dataclass
class LBFGSB_Parameters(OptimizerParameters):
    """Hyperparameters for LBFGSB optimizer"""

    ftol: float = 0.0
    gtol: float = 0.0


class LBFGSB(SciPyMinimizeOptimizer):
    """L-BFGS-B"""

    method: str = "L-BFGS-B"
    parameter: LBFGSB_Parameters = LBFGSB_Parameters()
