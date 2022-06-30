from typing import Any
import numpy as np
from f3dasm.src.designofexperiments import DoE
from f3dasm.src.simulation import Function
import pygmo as pg


class PygmoProblem:
    """Convert a testproblem from problemset to a pygmo object"""

    def __init__(self, design: DoE, func: Function, seed: Any or int = None):
        self.design = design
        self.func = func
        self.seed = seed

        if seed:
            pg.set_global_rng_seed(seed)

    def fitness(self, x: np.ndarray) -> np.ndarray:
        """Pygmo representation of returning the objective value of a function"""
        return self.func.eval(x).ravel()  # pygmo doc: should output 1D numpy array

    def batch_fitness(self, x: np.ndarray) -> np.ndarray:
        """Pygmo representation of returning multiple objective values of a function"""
        return self.fitness(x)

    def get_bounds(self) -> tuple:
        """Box-constrained boundaries of the problem. Necessary for pygmo library"""
        return (
            [
                parameter.lower_bound
                for parameter in self.design.get_continuous_parameters()
            ],
            [
                parameter.upper_bound
                for parameter in self.design.get_continuous_parameters()
            ],
        )

    def gradient(self, x: np.ndarray):
        return pg.estimate_gradient(lambda x: self.fitness(x), x)


# class Optimizer:
#     def __init__(self):
#         pass

#     def run(self, iterations: int):
#         for i in range(iterations):
#             pop = self.algorithm.evolve(pop)
#             xx = np.r_[xx, pop.get_x()]
#             yy = np.r_[yy, pop.get_f()]
