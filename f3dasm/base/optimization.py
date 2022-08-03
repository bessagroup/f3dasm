from dataclasses import dataclass, field
from typing import Any, Mapping, Optional
from f3dasm.base.data import Data
from f3dasm.base.simulation import Function

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
    seed: int or None = None
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
        updated_defaults.update(self.hyperparameters)
        self.hyperparameters = updated_defaults

    def set_algorithm(self) -> None:
        pass

    def update_step(self, function: Function) -> None:
        raise NotImplementedError(
            "You should implement an update step for your algorithm!"
        )

    def iterate(self, iterations: int, function: Function) -> None:
        for _ in range(iterations):
            self.update_step(function=function)

    def extract_data(self) -> Data:
        return self.data
