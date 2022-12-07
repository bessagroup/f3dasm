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
class SimulatedAnnealing_Parameters(OptimizerParameters):
    """Hyperparameters for Simulated Annealing optimizer"""

    population: int = 30
    Ts: float = 10.0
    Tf: float = 0.1
    n_T_adj: int = 10
    n_range_adj: int = 10
    bin_size: int = 10
    start_range: float = 1.0


class SimulatedAnnealing(PygmoAlgorithm):
    "DifferentialEvolution optimizer implemented from pygmo"

    parameter: SimulatedAnnealing_Parameters = SimulatedAnnealing_Parameters()

    def set_algorithm(self):
        self.algorithm = pg.algorithm(
            pg.simulated_annealing(
                Ts=self.parameter.Ts,
                Tf=self.parameter.Tf,
                n_T_adj=self.parameter.n_T_adj,
                n_range_adj=self.parameter.n_range_adj,
                bin_size=self.parameter.bin_size,
                start_range=self.parameter.start_range,
                seed=self.seed,
            )
        )
