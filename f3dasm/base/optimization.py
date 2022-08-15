from copy import copy
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

from ..base.data import Data
from ..base.function import Function


@dataclass
class Optimizer:
    """Mainclass to inherit from to implement optimization algorithms

    Args:
        data (Data): Data-object
        hyperparamaters (dict): Dictionary with hyperparamaters (default is None)
        seed (int): seed to set the optimizer
        defaults (dict): Default hyperparameter arguments
    """

    data: Data
    hyperparameters: Optional[Mapping[str, Any]] = field(default_factory=dict)
    seed: int = np.random.randint(low=0, high=1e5)
    defaults: Optional[Mapping[str, Any]] = field(default_factory=dict)
    algorithm: Any = field(init=False)

    def __post_init__(self):
        if self.seed:
            self.set_seed(self.seed)

        self.init_parameters()
        self._set_hyperparameters()
        self.set_algorithm()

    @staticmethod
    def set_seed(seed: int) -> None:
        """Set the seed of the optimizer. Needs to be inherited."""
        pass

    def init_parameters(self) -> None:
        """Set the initialization parameters. This could be dynamic or static hyperparameters."""
        pass

    def _set_hyperparameters(self) -> None:
        """Overwrite the default hyperparameters by the given ones"""
        updated_defaults = self.defaults.copy()

        # Check if population argument is present. Otherwise set to 1
        if "population" not in updated_defaults:
            updated_defaults["population"] = 1

        updated_defaults.update(self.hyperparameters)
        self.hyperparameters = updated_defaults

    def set_algorithm(self) -> None:
        """If necessary, the algorithm needs to be set"""
        pass

    def update_step(self, function: Function) -> None:
        """One iteration of the algorithm. Adds the new input vector and resulting output immediately to the data attribute

        Args:
            function (Function): Objective function to evaluate

        """
        raise NotImplementedError("You should implement an update step for your algorithm!")

    def iterate(self, iterations: int, function: Function) -> None:
        """Calls the update_step function multiple times.

        Args:
            iterations (int): number of iterations
            function (Function): Objective function to evaluate
        """
        for _ in range(iterations):
            self.update_step(function=function)

    def extract_data(self) -> Data:
        """Returns a copy of the data"""
        return copy(self.data)

    def _force_bounds(self, x: np.ndarray, scale_bounds: np.ndarray) -> np.ndarray:
        x = np.where(x < scale_bounds[:, 0], scale_bounds[:, 0], x)
        x = np.where(x > scale_bounds[:, 1], scale_bounds[:, 1], x)
        return x
