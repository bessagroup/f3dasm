from dataclasses import dataclass
from typing import Any
import numpy as np
import pygmo as pg

from ..optimization.hyperparameters import CMAES_Parameters, PSO_Parameters, SGA_Parameters, XNES_Parameters

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
        return self.func(x).ravel()  # pygmo doc: should output 1D numpy array

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
                design=self.data.design,
                func=function,
                seed=self.seed,
            )
        )

        # Construct the population
        pop = pg.population(prob, size=self.parameter.population)

        # Set the population to the latest datapoints
        pop_x = self.data.get_input_data().iloc[-self.parameter.population :].to_numpy()
        pop_fx = self.data.get_output_data().iloc[-self.parameter.population :].to_numpy()

        for index, (x, fx) in enumerate(zip(pop_x, pop_fx)):
            pop.set_xf(index, x, fx)

        # Iterate one step
        pop = self.algorithm.evolve(pop)

        # Add new population to data
        self.data.add_numpy_arrays(input=pop.get_x(), output=pop.get_f())


class CMAES(PygmoAlgorithm):
    """Covariance Matrix Adaptation Evolution Strategy optimizer implemented from pygmo"""

    parameter: CMAES_Parameters = CMAES_Parameters()

    def set_algorithm(self):
        self.algorithm = pg.algorithm(
            pg.cmaes(
                gen=self.parameter.gen,
                memory=self.parameter.memory,
                seed=self.seed,
                force_bounds=self.parameter.force_bounds,
            )
        )


class PSO(PygmoAlgorithm):
    "Particle Swarm Optimization (Generational) optimizer implemented from pygmo"

    parameter: PSO_Parameters = PSO_Parameters()

    def set_algorithm(self):
        self.algorithm = pg.algorithm(
            pg.pso_gen(
                gen=self.parameter.gen,
                memory=self.parameter.memory,
                seed=self.seed,
            )
        )


class SGA(PygmoAlgorithm):
    """Simple Genetic Algorithm optimizer implemented from pygmo"""

    parameter: SGA_Parameters = SGA_Parameters()

    def set_algorithm(self):
        self.algorithm = pg.algorithm(
            pg.sga(
                gen=self.parameter.gen,
                cr=self.parameter.cr,
                eta_c=self.parameter.eta_c,
                m=self.parameter.m,
                param_m=self.parameter.param_m,
                param_s=self.parameter.param_s,
                crossover=self.parameter.crossover,
                mutation=self.parameter.mutation,
                selection=self.parameter.selection,
                seed=self.seed,
            )
        )


class XNES(PygmoAlgorithm):
    """XNES optimizer implemented from pygmo"""

    parameter: XNES_Parameters = XNES_Parameters()

    def set_algorithm(self):
        self.algorithm = pg.algorithm(
            pg.xnes(
                gen=self.parameter.gen,
                eta_mu=self.parameter.eta_mu,
                eta_sigma=self.parameter.eta_sigma,
                eta_b=self.parameter.eta_b,
                sigma0=self.parameter.sigma0,
                ftol=self.parameter.ftol,
                xtol=self.parameter.xtol,
                memory=self.parameter.memory,
                force_bounds=self.parameter.force_bounds,
                seed=self.seed,
            )
        )
