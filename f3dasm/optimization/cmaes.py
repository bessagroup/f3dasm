from dataclasses import dataclass
import pygmo as pg

from .adapters.pygmo_implementations import PygmoAlgorithm
from ..base.optimization import OptimizerParameters


@dataclass
class CMAES_Parameters(OptimizerParameters):
    """Hyperparameters for CMAES optimizer"""

    population: int = 30
    gen: int = 1
    memory: bool = True


class CMAES(PygmoAlgorithm):
    """Covariance Matrix Adaptation Evolution Strategy optimizer implemented from pygmo"""

    parameter: CMAES_Parameters = CMAES_Parameters()

    def set_algorithm(self):
        self.algorithm = pg.algorithm(
            pg.cmaes(
                gen=self.parameter.gen,
                memory=self.parameter.memory,
                seed=self.seed,
                force_bounds=self.parameter.force_bounds,
            )
        )
