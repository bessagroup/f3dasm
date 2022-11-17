from copy import copy
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import autograd.numpy as np

from ..base.data import Data
from ..base.function import Function
from ..base.utils import _number_of_overiterations, _number_of_updates


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

    def set_data(self, data: Data):
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

    def _construct_model(self, function: Function):
        """Construct a model necessary for iteration with input of to be evaluated function

        Parameters
        ----------
        function
            function to be evaluated
        """
        pass

    def iterate(self, iterations: int, function: Function):
        """Calls the update_step function multiple times.

        Parameters
        ----------
        iterations
            number of iterations
        function
            objective function to evaluate
        """
        self._construct_model(function)

        self._check_number_of_datapoints()

        for _ in range(_number_of_updates(iterations, population=self.parameter.population)):
            self.update_step(function=function)

        # Remove overiterations
        self.data.remove_rows_bottom(_number_of_overiterations(iterations, population=self.parameter.population))
        print(f"Optimizing for {iterations} iterations with {self.get_name()}")

    def extract_data(self) -> Data:
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
