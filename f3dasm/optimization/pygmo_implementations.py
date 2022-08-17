from dataclasses import dataclass
from typing import Any
import numpy as np
import pygmo as pg

from ..base.design import DesignSpace
from ..base.optimization import Optimizer
from ..base.function import Function


@dataclass
class PygmoProblem:
    """Convert a testproblem from problemset to a pygmo object"""

    design: DesignSpace
    func: Function
    seed: Any or int = None

    def __post_init__(self):
        if self.seed:
            pg.set_global_rng_seed(self.seed)

    def fitness(self, x: np.ndarray) -> np.ndarray:
        """Pygmo representation of returning the objective value of a function"""
        return self.func.__call__(x).ravel()  # pygmo doc: should output 1D numpy array

    def batch_fitness(self, x: np.ndarray) -> np.ndarray:
        """Pygmo representation of returning multiple objective values of a function"""
        return self.fitness(x)

    def get_bounds(self) -> tuple:
        """Box-constrained boundaries of the problem. Necessary for pygmo library"""
        return (
            [parameter.lower_bound for parameter in self.design.get_continuous_input_parameters()],
            [parameter.upper_bound for parameter in self.design.get_continuous_input_parameters()],
        )

    def gradient(self, x: np.ndarray):
        # return pg.estimate_gradient(lambda x: self.fitness(x), x)
        return self.func.dfdx(x)


@dataclass
class PygmoAlgorithm(Optimizer):
    """Wrapper around the pygmo algorithm class"""

    @staticmethod
    def set_seed(seed: int) -> None:
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
        pop_x = self.data.get_input_data().iloc[-self.hyperparameters["population"] :].to_numpy()
        pop_fx = self.data.get_output_data().iloc[-self.hyperparameters["population"] :].to_numpy()

        for index, (x, fx) in enumerate(zip(pop_x, pop_fx)):
            pop.set_xf(index, x, fx)

        # Iterate one step
        pop = self.algorithm.evolve(pop)

        # Add new population to data
        self.data.add_numpy_arrays(input=pop.get_x(), output=pop.get_f())


class CMAES(PygmoAlgorithm):
    """Covariance Matrix Adaptation Evolution Strategy optimizer implemented from pygmo"""

    def init_parameters(self):
        # Default hyperparameters
        self.defaults = {
            "gen": 1,
            "memory": True,
            "force_bounds": True,
            "population": 30,
        }

    def set_algorithm(self):
        self.algorithm = pg.algorithm(
            pg.cmaes(
                gen=self.hyperparameters["gen"],
                memory=self.hyperparameters["memory"],
                seed=self.seed,
                force_bounds=self.hyperparameters["force_bounds"],
            )
        )


class PSO(PygmoAlgorithm):
    "Particle Swarm Optimization (Generational) optimizer implemented from pygmo"

    def init_parameters(self):
        # Default hyperparameters
        self.defaults = {
            "gen": 1,
            "memory": True,
            "population": 30,
        }

    def set_algorithm(self):
        self.algorithm = pg.algorithm(
            pg.pso_gen(
                gen=self.hyperparameters["gen"],
                memory=self.hyperparameters["memory"],
                seed=self.seed,
            )
        )


class SGA(PygmoAlgorithm):
    """Simple Genetic Algorithm optimizer implemented from pygmo"""

    def init_parameters(self):
        # Default hyperparameters
        self.defaults = {
            "gen": 1,
            "cr": 0.9,
            "eta_c": 1.0,
            "m": 0.02,
            "param_m": 1.0,
            "param_s": 2,
            "crossover": "exponential",
            "mutation": "polynomial",
            "selection": "tournament",
            "population": 30,
        }

    def set_algorithm(self):
        self.algorithm = pg.algorithm(
            pg.sga(
                gen=self.hyperparameters["gen"],
                cr=self.hyperparameters["cr"],
                eta_c=self.hyperparameters["eta_c"],
                m=self.hyperparameters["m"],
                param_m=self.hyperparameters["param_m"],
                param_s=self.hyperparameters["param_s"],
                crossover=self.hyperparameters["crossover"],
                mutation=self.hyperparameters["mutation"],
                selection=self.hyperparameters["selection"],
                seed=self.seed,
            )
        )


class XNES(PygmoAlgorithm):
    """XNES optimizer implemented from pygmo"""

    def init_parameters(self):
        self.defaults = {
            "gen": 1,
            "eta_mu": -1,
            "eta_sigma": -1,
            "eta_b": -1,
            "sigma0": -1,
            "ftol": 1e-06,
            "xtol": 1e-06,
            "memory": True,
            "force_bounds": True,
            "population": 30,
        }

    def set_algorithm(self):
        self.algorithm = pg.algorithm(
            pg.xnes(
                gen=self.hyperparameters["gen"],
                eta_mu=self.hyperparameters["eta_mu"],
                eta_sigma=self.hyperparameters["eta_sigma"],
                eta_b=self.hyperparameters["eta_b"],
                sigma0=self.hyperparameters["sigma0"],
                ftol=self.hyperparameters["ftol"],
                xtol=self.hyperparameters["xtol"],
                memory=self.hyperparameters["memory"],
                force_bounds=self.hyperparameters["force_bounds"],
                seed=self.seed,
            )
        )
