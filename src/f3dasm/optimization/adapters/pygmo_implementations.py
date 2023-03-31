#                                                                       Modules
# =============================================================================

# Standard
from dataclasses import dataclass
from typing import Any, List, Tuple

# Third-party core
import autograd.numpy as np

# Locals
from ..._imports import try_import
from .._protocol import DesignSpace, Function
from ..optimizer import Optimizer

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
class _PygmoProblem:
    """Convert a testproblem from the problemset to pygmo object

    Parameters
    ----------
    design
        designspace to be used
    func
        function to be evaluated
    seed
        seed for the random number generator
        _description_
    """
    design: DesignSpace
    func: Function
    seed: Any or int = None

    def __post_init__(self):
        if self.seed:
            pg.set_global_rng_seed(self.seed)

    def fitness(self, x: np.ndarray) -> np.ndarray:
        """Pygmo representation of returning the objective value of a function

        Parameters
        ----------
        x
            input vector

        Returns
        -------
            fitness
        """
        return self.func(x).ravel()  # pygmo doc: should output 1D numpy array

    def batch_fitness(self, x: np.ndarray) -> np.ndarray:
        """Pygmo representation of returning multiple objective values of a function

        Parameters
        ----------
        x
            input vectors

        Returns
        -------
            fitnesses
        """
        # Pygmo representation of returning multiple objective values of a function
        return self.fitness(x)

    def get_bounds(self) -> Tuple[List[float], List[float]]:
        """Box-constrained boundaries of the problem. Necessary for pygmo library

        Returns
        -------
            box constraints
        """
        return (
            [parameter.lower_bound for parameter in self.design.get_continuous_input_parameters()],
            [parameter.upper_bound for parameter in self.design.get_continuous_input_parameters()],
        )

    def gradient(self, x: np.ndarray):
        """Gradient in pygmo accepted format

        Parameters
        ----------
        x
            input vector

        Returns
        -------
            gradient
        """
        # return pg.estimate_gradient(lambda x: self.fitness(x), x)
        return self.func.dfdx(x)


@dataclass
class PygmoAlgorithm(Optimizer):
    """Wrapper around the pygmo algorithm class

    Parameters
    ----------
    data
        ExperimentData-object
    hyperparameters
        Dictionary with hyperparameters
    seed
        seed to set the optimizer
    defaults
        Default hyperparameter arguments
    """

    @staticmethod
    def _check_imports():
        _imports.check()

    @staticmethod
    def set_seed(seed: int):
        """Set the seed for pygmo

        Parameters
        ----------
        seed
            seed for the random number generator
        """
        pg.set_global_rng_seed(seed=seed)

    def update_step(self, function: Function) -> Tuple[np.ndarray, np.ndarray]:
        """Update step of the algorithm

        Parameters
        ----------
        function
            function to be evaluated

        Returns
        -------
            tuple of updated input parameters (x) and objecti value (y)
        """
        # Construct the PygmoProblem
        prob = pg.problem(
            _PygmoProblem(
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
