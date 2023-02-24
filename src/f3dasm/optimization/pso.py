#                                                                       Modules
# =============================================================================

# Standard
from dataclasses import dataclass

# Third-party
import pygmo as pg

# Locals
from .optimizer import OptimizerParameters
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
class PSO_Parameters(OptimizerParameters):
    """Hyperparameters for PSO optimizer"""

    population: int = 30
    eta1: float = 2.05
    eta2: float = 2.05


class PSO(PygmoAlgorithm):
    "Particle Swarm Optimization (Generational) optimizer implemented from pygmo"

    parameter: PSO_Parameters = PSO_Parameters()

    def set_algorithm(self):
        self.algorithm = pg.algorithm(
            pg.pso_gen(
                gen=1,
                memory=True,
                seed=self.seed,
                eta1=self.parameter.eta1,
                eta2=self.parameter.eta2,
            )
        )
