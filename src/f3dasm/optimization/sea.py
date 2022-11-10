from dataclasses import dataclass

import pygmo as pg

from ..base.optimization import OptimizerParameters
from .adapters.pygmo_implementations import PygmoAlgorithm


@dataclass
class SEA_Parameters(OptimizerParameters):
    """Hyperparameters for SEA optimizer"""

    population: int = 30


class SEA(PygmoAlgorithm):
    """Simple Evolutionary Algorithm optimizer implemented from pygmo"""

    parameter: SEA_Parameters = SEA_Parameters()

    def set_algorithm(self):
        self.algorithm = pg.algorithm(
            pg.sea(
                gen=1,
                seed=self.seed,
            )
        )
