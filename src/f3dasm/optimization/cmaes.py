from dataclasses import dataclass

import pygmo as pg

from ..base.function import Function
from ..base.optimization import OptimizerParameters
from .adapters.pygmo_implementations import PygmoAlgorithm


@dataclass
class CMAES_Parameters(OptimizerParameters):
    """Hyperparameters for CMAES optimizer"""

    population: int = 30


class CMAES(PygmoAlgorithm):
    """Covariance Matrix Adaptation Evolution Strategy optimizer implemented from pygmo"""

    parameter: CMAES_Parameters = CMAES_Parameters()

    def set_algorithm(self):
        self.algorithm = pg.algorithm(
            pg.cmaes(
                gen=1,
                memory=True,
                seed=self.seed,
                force_bounds=self.parameter.force_bounds,
            )
        )
