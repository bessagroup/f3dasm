from dataclasses import dataclass

from ..base.optimization import OptimizerParameters
from .adapters.scipy_implementations import SciPyMinimizeOptimizer


# @dataclass
class CG_Parameters(OptimizerParameters):
    """_summary_

    Args:
        gtol (float): _description_ (Default = 0.0)
    """

    gtol: float = 0.0


class CG(SciPyMinimizeOptimizer):
    """CG"""

    method: str = "CG"
    parameter: CG_Parameters = CG_Parameters()
