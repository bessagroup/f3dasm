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
        self.set_hyperparameters()
        self.set_algorithm()

    @staticmethod
    def set_seed(seed: int) -> None:
        pass

    def set_data(self, data: Data) -> None:
        self.data = data

    def init_parameters(self) -> None:
        pass

    def set_hyperparameters(self) -> None:
        updated_defaults = self.defaults.copy()

        # Check if population argument is present
        if "population" not in updated_defaults:
            updated_defaults["population"] = 1

        updated_defaults.update(self.hyperparameters)
        self.hyperparameters = updated_defaults

    def set_algorithm(self) -> None:
        pass

    def update_step(self, function: Function) -> None:
        raise NotImplementedError("You should implement an update step for your algorithm!")

    def iterate(self, iterations: int, function: Function) -> None:
        for _ in range(iterations):
            self.update_step(function=function)

    def _force_bounds(self, x: np.ndarray, scale_bounds: np.ndarray) -> np.ndarray:
        x = np.where(x < scale_bounds[:, 0], scale_bounds[:, 0], x)
        x = np.where(x > scale_bounds[:, 1], scale_bounds[:, 1], x)
        return x

    def extract_data(self) -> Data:
        return copy(self.data)
