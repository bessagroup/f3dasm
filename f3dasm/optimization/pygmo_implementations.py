from typing import Any, Mapping, Optional
import numpy as np
import pygmo as pg

from f3dasm.base.data import Data
from f3dasm.base.designofexperiments import DesignSpace
from f3dasm.base.optimization import Optimizer
from f3dasm.base.simulation import Function


class PygmoProblem:
    """Convert a testproblem from problemset to a pygmo object"""

    def __init__(self, design: DesignSpace, func: Function, seed: Any or int = None):
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
        # return pg.estimate_gradient(lambda x: self.fitness(x), x)
        return self.func.dfdx(x)


class PygmoAlgorithm(Optimizer):
    @staticmethod
    def set_seed(seed: int):
        pg.set_global_rng_seed(seed=seed)

    def update_step(self, function: Function) -> None:

        # Construct the PygmoProblem
        prob = pg.problem(
            PygmoProblem(
                design=self.data.designspace,
                func=function,
                seed=self.seed,
            )
        )

        # Construct the population
        pop = pg.population(prob, size=self.hyperparameters["population"])

        # Set the population to the latest datapoints
        pop_x = (
            self.data.get_input_data()
            .iloc[-self.hyperparameters["population"] :]
            .to_numpy()
        )
        pop_fx = (
            self.data.get_output_data()
            .iloc[-self.hyperparameters["population"] :]
            .to_numpy()
        )

        for index, (x, fx) in enumerate(zip(pop_x, pop_fx)):
            pop.set_xf(index, x, fx)

        # Iterate one step
        pop = self.algorithm.evolve(pop)

        # Add new population to data
        self.data.add_numpy_arrays(input=pop.get_x(), output=pop.get_f())


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
                seed=seed,
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
                seed=seed,
            )
        )

        super().__init__(data=data, hyperparameters=self.hyperparameters, seed=seed)
