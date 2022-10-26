from dataclasses import dataclass
import pygmo as pg

from .adapters.pygmo_implementations import PygmoAlgorithm
from ..base.optimization import OptimizerParameters


@dataclass
class SEA_Parameters(OptimizerParameters):
    """Hyperparameters for SEA optimizer"""

    population: int = 30
    gen: int = 1


class SEA(PygmoAlgorithm):
    """Simple Evolutionary Algorithm optimizer implemented from pygmo"""

    parameter: SEA_Parameters = SEA_Parameters()

    def set_algorithm(self):
        self.algorithm = pg.algorithm(
            pg.sea(
                gen=self.parameter.gen,
                seed=self.seed,
            )
        )
