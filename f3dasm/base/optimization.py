from typing import Any, Mapping, Optional
from f3dasm.base.data import Data
from f3dasm.base.simulation import Function


class Optimizer:
    def __init__(
        self,
        data: Data,
        hyperparameters: Optional[Mapping[str, Any]] = None,
        seed: int or None = None,
    ):

        self.data = data
        self.seed = seed

        self.init_parameters()
        self.set_hyperparameters(hyperparameters)

        if self.seed:
            self.set_seed(seed)

    @staticmethod
    def set_seed(seed: int):
        pass

    def set_data(self, data: Data):
        self.data = data

    def init_parameters(self):
        pass

    def set_hyperparameters(self, hyperparameters: Optional[Mapping[str, Any]]):
        if hyperparameters is None:
            hyperparameters = {}

        if not hasattr(self, "defaults"):
            self.defaults = {}

        self.hyperparameters = self.defaults.copy()
        self.hyperparameters.update(hyperparameters)

    def update_step(self, function: Function):
        raise NotImplementedError()

    def iterate(self, iterations: int, function: Function) -> None:
        for _ in range(iterations):
            self.update_step(function=function)

    def extract_data(self):
        return self.data
