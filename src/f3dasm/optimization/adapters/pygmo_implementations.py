#                                                                       Modules
# =============================================================================

# Standard
from dataclasses import dataclass
from typing import Any, Tuple

# Third-party
import autograd.numpy as np
import pygmo as pg

# Locals
from ...base.design import DesignSpace
from ...base.function import Function
from ...base.optimization import Optimizer

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


@dataclass
class PygmoProblem:
    """Convert a testproblem from problemset to a pygmo object

    :param design: Designspace
    :param func: function to be evaluated
    :param seed: seed for the random number generator
    """

    design: DesignSpace
    func: Function
    seed: Any or int = None

    def __post_init__(self):
        if self.seed:
            pg.set_global_rng_seed(self.seed)

    def fitness(self, x: np.ndarray) -> np.ndarray:
        """Pygmo representation of returning the objective value of a function

        :param x: input vector
        :return: fitness
        """
        return self.func(x).ravel()  # pygmo doc: should output 1D numpy array

    def batch_fitness(self, x: np.ndarray) -> np.ndarray:
        """Pygmo representation of returning multiple objective values of a function

        :param x: input vectors
        :return: fitnesses
        """
        # Pygmo representation of returning multiple objective values of a function
        return self.fitness(x)

    def get_bounds(self) -> tuple:
        """Box-constrained boundaries of the problem. Necessary for pygmo library

        :return: box constraints
        """
        # Box-constrained boundaries of the problem. Necessary for pygmo library
        return (
            [parameter.lower_bound for parameter in self.design.get_continuous_input_parameters()],
            [parameter.upper_bound for parameter in self.design.get_continuous_input_parameters()],
        )

    def gradient(self, x: np.ndarray):
        """Gradient in pygmo accepted format

        :param x: input vector
        :return: gradient
        """
        # return pg.estimate_gradient(lambda x: self.fitness(x), x)
        return self.func.dfdx(x)


@dataclass
class PygmoAlgorithm(Optimizer):
    """Wrapper around the pygmo algorithm class

    :param data: Data-object
    :param hyperparameters: Dictionary with hyperparamaters
    :param seed: seed to set the optimizer
    :param defaults: Default hyperparameter arguments
    """

    @staticmethod
    def set_seed(seed: int):
        """Set the seed for pygmo

        :param seed: seed for the random number generator
        """
        pg.set_global_rng_seed(seed=seed)

    def update_step(self, function: Function) -> Tuple[np.ndarray, np.ndarray]:
        """Update step of the algorithm

        :param function: function to be evaluated
        """

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
        pop_x = self.data.get_input_data(
        ).iloc[-self.parameter.population:].to_numpy()
        pop_fx = self.data.get_output_data(
        ).iloc[-self.parameter.population:].to_numpy()

        for index, (x, fx) in enumerate(zip(pop_x, pop_fx)):
            pop.set_xf(index, x, fx)

        # Iterate one step
        pop = self.algorithm.evolve(pop)

        return pop.get_x(), pop.get_f()
