from dataclasses import dataclass
import pygmo as pg


from ..base.optimization import OptimizerParameters
from .adapters.pygmo_implementations import PygmoAlgorithm


@dataclass
class SADE_Parameters(OptimizerParameters):
    """Hyperparameters for Self-adaptive Differential Evolution optimizer"""

    population: int = 30
    gen: int = 1
    variant: int = 2
    variant_adptv: int = 1
    ftol: float = 0.0
    xtol: float = 0.0
    memory: bool = True


class SADE(PygmoAlgorithm):
    "Self-adaptive Differential Evolution optimizer implemented from pygmo"

    parameter: SADE_Parameters = SADE_Parameters()

    def set_algorithm(self):
        self.algorithm = pg.algorithm(
            pg.sade(
                gen=self.parameter.gen,
                variant=self.parameter.variant,
                variant_adptv=self.parameter.variant_adptv,
                ftol=self.parameter.ftol,
                xtol=self.parameter.xtol,
                memory=self.parameter.memory,
                seed=self.seed,
            )
        )
