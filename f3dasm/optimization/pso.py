from dataclasses import dataclass
import pygmo as pg


from ..base.optimization import OptimizerParameters
from .adapters.pygmo_implementations import PygmoAlgorithm


@dataclass
class PSO_Parameters(OptimizerParameters):
    """Hyperparameters for PSO optimizer"""

    population: int = 30
    gen: int = 1
    memory: bool = True


class PSO(PygmoAlgorithm):
    "Particle Swarm Optimization (Generational) optimizer implemented from pygmo"

    parameter: PSO_Parameters = PSO_Parameters()

    def set_algorithm(self):
        self.algorithm = pg.algorithm(
            pg.pso_gen(
                gen=self.parameter.gen,
                memory=self.parameter.memory,
                seed=self.seed,
            )
        )
