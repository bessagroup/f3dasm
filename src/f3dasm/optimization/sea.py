#                                                                       Modules
# =============================================================================

# Standard
from dataclasses import dataclass
from typing import List

# Locals
from .._imports import try_import
from .adapters.pygmo_implementations import PygmoAlgorithm
from .optimizer import OptimizerParameters

# Third-party extension
with try_import('optimization') as _imports:
    import pygmo as pg


#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


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

    def get_info(self) -> List[str]:
        return ['Fast', 'Global', 'Derivative-Free', 'Population-Based']
