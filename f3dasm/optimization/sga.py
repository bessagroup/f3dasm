from dataclasses import dataclass
import pygmo as pg

from .adapters.pygmo_implementations import PygmoAlgorithm
from ..base.optimization import OptimizerParameters


@dataclass
class SGA_Parameters(OptimizerParameters):
    """Hyperparameters for SGA optimizer"""

    gen: int = 1
    cr: float = 0.9
    eta_c: float = 1.0
    m: float = 0.02
    param_m: float = 1.0
    param_s: float = 2
    crossover: str = "exponential"
    mutation: str = "polynomial"
    selection: str = "tournament"
    population: int = 30


class SGA(PygmoAlgorithm):
    """Simple Genetic Algorithm optimizer implemented from pygmo"""

    parameter: SGA_Parameters = SGA_Parameters()

    def set_algorithm(self):
        self.algorithm = pg.algorithm(
            pg.sga(
                gen=self.parameter.gen,
                cr=self.parameter.cr,
                eta_c=self.parameter.eta_c,
                m=self.parameter.m,
                param_m=self.parameter.param_m,
                param_s=self.parameter.param_s,
                crossover=self.parameter.crossover,
                mutation=self.parameter.mutation,
                selection=self.parameter.selection,
                seed=self.seed,
            )
        )
