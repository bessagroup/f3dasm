from dataclasses import dataclass
import pygmo as pg

from .adapters.pygmo_implementations import PygmoAlgorithm
from ..base.optimization import OptimizerParameters


@dataclass
class XNES_Parameters(OptimizerParameters):
    """Hyperparameters for XNES optimizer

    Args:
        population (int): _description_ (Default = 30)
        eta_mu (float): _description_ (Default = -1.0)
        eta_sigma (float): _description_ (Default = -1.0)
        eta_b (float): _description_ (Default = -1.0)
        sigma0 (float): _description_ (Default = -1.0)
        ftol (float): _description_ (Default = 1e-06)
        xtol (float): _description_ (Default = 1e-06)
    """

    population: int = 30
    eta_mu: float = -1.0
    eta_sigma: float = -1.0
    eta_b: float = -1.0
    sigma0: float = -1.0
    ftol: float = 1e-06
    xtol: float = 1e-06


class XNES(PygmoAlgorithm):
    """XNES optimizer implemented from pygmo"""

    parameter: XNES_Parameters = XNES_Parameters()

    def set_algorithm(self):
        self.algorithm = pg.algorithm(
            pg.xnes(
                gen=1,
                eta_mu=self.parameter.eta_mu,
                eta_sigma=self.parameter.eta_sigma,
                eta_b=self.parameter.eta_b,
                sigma0=self.parameter.sigma0,
                ftol=self.parameter.ftol,
                xtol=self.parameter.xtol,
                memory=True,
                force_bounds=self.parameter.force_bounds,
                seed=self.seed,
            )
        )
