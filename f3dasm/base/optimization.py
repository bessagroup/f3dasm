from copy import copy
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import autograd.numpy as np

from ..base.data import Data
from ..base.function import Function


@dataclass
class OptimizerParameters:
    population: int = 1
    force_bounds: bool = True


@dataclass
class Optimizer:
    """Mainclass to inherit from to implement optimization algorithms

    Args:
        data (Data): Data-object
        hyperparameters (dict): Dictionary with hyperparamaters (default is None)
        seed (int): seed to set the optimizer
        defaults (dict): Default hyperparameter arguments
    """

    data: Data
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
    def set_seed(seed: int) -> None:
        """Set the seed of the optimizer. Needs to be inherited."""
        pass

    def init_parameters(self) -> None:
        """Set the initialization parameters. This could be dynamic or static hyperparameters."""
        pass

    def set_algorithm(self) -> None:
        """If necessary, the algorithm needs to be set"""
        pass

    def update_step(self, function: Function) -> None:
        """One iteration of the algorithm. Adds the new input vector and resulting output immediately to the data attribute

        Args:
            function (Function): Objective function to evaluate

        """
        raise NotImplementedError("You should implement an update step for your algorithm!")

    def set_data(self, data: Data) -> None:
        self.data = data

    def _set_hyperparameters(self) -> None:
        """New way of setting hyperparameters with dedicated class"""
        # if isinstance(self.hyperparameters, dict):
        #     # Create instance of the specific hyperparameter class
        self.parameter.__init__(**self.hyperparameters)

    def _check_number_of_datapoints(self) -> None:
        """Check if available data => population size"""
        if self.data.get_number_of_datapoints() < self.parameter.population:
            raise ValueError(
                f"There are {self.data.get_number_of_datapoints()} datapoints available, need {self.parameter.population} for update step!"
            )
        return

    def iterate(self, iterations: int, function: Function) -> None:
        """Calls the update_step function multiple times.

        Args:
            iterations (int): number of iterations
            function (Function): Objective function to evaluate
        """

        self._check_number_of_datapoints()

        for _ in range(self._number_of_updates(iterations)):
            self.update_step(function=function)

        # Remove overiterations
        self.data.remove_rows_bottom(self._number_of_overiterations(iterations))

    def _number_of_updates(self, iterations: int) -> int:
        return iterations // self.parameter.population + (iterations % self.parameter.population > 0)

    def _number_of_overiterations(self, iterations: int) -> int:
        overiterations: int = iterations % self.parameter.population
        if overiterations == 0:
            return overiterations
        else:
            return self.parameter.population - overiterations

    def extract_data(self) -> Data:
        """Returns a copy of the data"""
        return copy(self.data)

    def get_name(self) -> str:
        return self.__class__.__name__
