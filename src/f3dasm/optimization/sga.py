#                                                                       Modules
# =============================================================================

# Standard
from dataclasses import dataclass

# Third-party
import pygmo as pg

# Locals
from ..base.optimization import OptimizerParameters
from .adapters.pygmo_implementations import PygmoAlgorithm

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


@dataclass
class SGA_Parameters(OptimizerParameters):
    """Hyperparameters for SGA optimizer"""

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
                gen=1,
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
