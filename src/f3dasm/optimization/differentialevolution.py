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
class DifferentialEvolution_Parameters(OptimizerParameters):
    """Hyperparameters for DifferentialEvolution optimizer

    Args:
        population (int): _description_ (Default = 30)
        F (float): _description_ (Default = 0.8)
        CR (float): _description_ (Default = 0.9)
        variant (int): _description_ (Default = 2)
        ftol (float): _description_ (Default = 0.0)
        xtol (float): _description_ (Default = 0.0)
    """

    population: int = 30
    F: float = 0.8
    CR: float = 0.9
    variant: int = 2
    ftol: float = 0.0
    xtol: float = 0.0


class DifferentialEvolution(PygmoAlgorithm):
    "DifferentialEvolution optimizer implemented from pygmo"

    parameter: DifferentialEvolution_Parameters = DifferentialEvolution_Parameters()

    def set_algorithm(self):
        self.algorithm = pg.algorithm(
            pg.de(
                gen=1,
                F=self.parameter.F,
                CR=self.parameter.CR,
                variant=self.parameter.variant,
                ftol=self.parameter.ftol,
                xtol=self.parameter.xtol,
                seed=self.seed,
            )
        )
