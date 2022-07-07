from typing import Any
from f3dasm.base.data import Data

from f3dasm.base.optimization import PygmoAlgorithm
import pygmo as pg


class CMAES(PygmoAlgorithm):
    def __init__(self, data: Data, seed: int or Any = None, population: int = 30):
        algorithm = pg.cmaes(gen=1, memory=True, seed=seed, force_bounds=True)

        super().__init__(
            data=data,
            algorithm=pg.algorithm(algorithm),
            seed=seed,
            population=population,
        )


class PSO(PygmoAlgorithm):
    def __init__(self, data: Data, seed: int or Any = None, population: int = 30):
        algorithm = pg.pso_gen(gen=1, memory=True, seed=seed)

        super().__init__(
            data=data,
            algorithm=pg.algorithm(algorithm),
            seed=seed,
            population=population,
        )
