from typing import Any, Mapping, Optional
from f3dasm.base.data import Data

from f3dasm.base.optimization import PygmoAlgorithm
import pygmo as pg


class CMAES(PygmoAlgorithm):
    def __init__(
        self,
        data: Data,
        hyperparameters: Optional[Mapping[str, Any]] = None,
        seed: int or None = None,
    ):

        # Default hyperparameters
        self.defaults = {
            "gen": 1,
            "memory": True,
            "force_bounds": True,
            "population": 30,
        }

        self.set_hyperparameters(hyperparameters)

        self.algorithm = pg.algorithm(
            pg.cmaes(
                gen=self.hyperparameters["gen"],
                memory=self.hyperparameters["memory"],
                seed=self.seed,
                force_bounds=self.hyperparameters["force_bounds"],
            )
        )

        super().__init__(data=data, hyperparameters=self.hyperparameters, seed=seed)


class PSO(PygmoAlgorithm):
    def __init__(
        self,
        data: Data,
        hyperparameters: Optional[Mapping[str, Any]] = None,
        seed: int or None = None,
    ):
        # Default hyperparameters
        self.defaults = {
            "gen": 1,
            "memory": True,
            "population": 30,
        }
        self.set_hyperparameters(hyperparameters)

        self.algorithm = pg.algorithm(
            pg.pso_gen(
                gen=self.hyperparameters["gen"],
                memory=self.hyperparameters["memory"],
                seed=self.seed,
            )
        )

        super().__init__(data=data, hyperparameters=self.hyperparameters, seed=seed)
