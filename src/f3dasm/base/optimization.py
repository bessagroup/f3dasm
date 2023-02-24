from copy import copy
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, List

import autograd.numpy as np

from ..design import ExperimentData
from ..base.function import Function, MultiFidelityFunction


@dataclass
class OptimizerParameters:
    """Interface of a continuous benchmark function

    Parameters
    ----------
    population : int
        population of the optimizer update step
    force_bounds : bool
        force the optimizer to not exceed the boundaries of the designspace
    """

    population: int = 1
    force_bounds: bool = True


@dataclass
class Optimizer:
    """Mainclass to inherit from to implement optimization algorithms

    Parameters
    -------
    data : f3dasm.base.data.Data
        Data-object
    hyperparameters : dict
        Dictionary with hyperparameteres
    seed : int
        seed to set the optimizer
    defaults : OptimizerParameters
        Default hyperparameter arguments

    Raises
    ------
    NotImplementedError
        When no update step is implemented
    ValueError
        When number of datapoints is lower than the population
    """

    data: ExperimentData or List[ExperimentData]
    hyperparameters: Optional[Mapping[str, Any]] = field(default_factory=dict)
    seed: int = np.random.randint(low=0, high=1e5)
    algorithm: Any = field(init=False)
    parameter: OptimizerParameters = field(init=False)

    def __post_init__(self):
        if self.seed:
            self.set_seed(self.seed)

        self.init_parameters()
        self._set_hyperparameters()
        self.set_algorithm()

    @staticmethod
    def _force_bounds(x: np.ndarray, scale_bounds: np.ndarray) -> np.ndarray:
        x = np.where(x < scale_bounds[:, 0], scale_bounds[:, 0], x)
        x = np.where(x > scale_bounds[:, 1], scale_bounds[:, 1], x)
        return x

    @staticmethod
    def set_seed(seed: int):
        """Set the seed of the optimizer. Needs to be inherited

        Parameters
        ----------
        seed
            seed for the random number generator
        """        
        pass

    def init_parameters(self):
        """Set the initialization parameters. This could be dynamic or static hyperparameters."""
        pass

    def set_algorithm(self):
        """If necessary, the algorithm needs to be set"""
        pass

    def update_step(self, function: Function):
        """One iteration of the algorithm. Adds the new input vector and resulting output immediately to the data attribute

        Parameters
        ----------
        function
            Objective function to evaluate

        Raises
        ------
        NotImplementedError
            You should implement an update step for your algorithm!
        """
        raise NotImplementedError("You should implement an update step for your algorithm!")

    def update_step_mf(self, multifidelity_function: MultiFidelityFunction, iteration: int,) -> None:
        """One iteration of the algorithm. Adds the new input vector and resulting output immediately to the data attribute

        Args:
            multifidelity_function (MultiFidelityFunction): Objective function to evaluate
            iteration (int): iteration

        """
        raise NotImplementedError("You should implement an update step for your algorithm!")

    def set_data(self, data: ExperimentData):
        """Overwrite the data attribute

        Parameters
        ----------
        data
            data object
        """        
        self.data = data

    def _set_hyperparameters(self):
        """New way of setting hyperparameters with dedicated class"""
        # if isinstance(self.hyperparameters, dict):
        #     # Create instance of the specific hyperparameter class
        self.parameter.__init__(**self.hyperparameters)

    def _check_number_of_datapoints(self):
        """Check if available data => population size"""
        if self.data.get_number_of_datapoints() < self.parameter.population:
            raise ValueError(
                f"There are {self.data.get_number_of_datapoints()} datapoints available, need {self.parameter.population} for update step!"
            )
        return

    def iterate(self, iterations: int, function: Function):
        """Calls the update_step function multiple times.

        Parameters
        ----------
        iterations
            number of iterations
        function
            objective function to evaluate
        """
        self._check_number_of_datapoints()

        for _ in range(self._number_of_updates(iterations)):
            self.update_step(function=function)

        # Remove overiterations
        self.data.remove_rows_bottom(self._number_of_overiterations(iterations))

    def iterate_mf(self, iterations: int, multifidelity_function: MultiFidelityFunction, budget: float,) -> None:
        """Calls the update_step function multiple times.

        Args:
            iterations (int): number of iterations
            mffunction (MultiFidelityFunction): Multi-fidelity function containing fidelity functions to evaluate
            budget (float): optimization algorithm terminates when this budget threshold is exceeded
        """

        # self._check_number_of_datapoints()
        cumulative_cost = 0
        for _ in range(self._number_of_updates(iterations)):
            if cumulative_cost < budget:
                print('iteration', _)#, self.data[-1].data)
                print('cumulative cost', float(cumulative_cost))
                self.update_step_mf(multifidelity_function=multifidelity_function, iteration=_)
                cumulative_cost += self.cost
            else:
                break

        # Remove overiterations
        # self.data[-1].remove_rows_bottom(self._number_of_overiterations(iterations))

    def _number_of_updates(self, iterations: int):
        return iterations // self.parameter.population + (iterations % self.parameter.population > 0)

    def _number_of_overiterations(self, iterations: int) -> int:
        overiterations: int = iterations % self.parameter.population
        if overiterations == 0:
            return overiterations
        else:
            return self.parameter.population - overiterations

    def extract_data(self) -> ExperimentData:        
        """Returns a copy of the data

        Returns
        -------
            copy of the data
        """
        return copy(self.data)

    def get_name(self) -> str:
        """Retrieve the name of the optimizers

        Returns
        -------
            name of the optimizer
        """        
        return self.__class__.__name__
