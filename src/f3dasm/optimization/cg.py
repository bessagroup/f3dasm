from dataclasses import dataclass

from ..base.optimization import OptimizerParameters
from .adapters.scipy_implementations import SciPyMinimizeOptimizer


# @dataclass
class CG_Parameters(OptimizerParameters):
    """CG Parameters"""

    gtol: float = 0.0


class CG(SciPyMinimizeOptimizer):
    """CG"""

    method: str = "CG"
    parameter: CG_Parameters = CG_Parameters()
