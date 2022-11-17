from abc import ABC
from copy import copy

from ..base.data import Data
from ..base.function import Function
from ..base.optimization import Optimizer


class Strategy(ABC):
    def get_optimizer(iteration_number: int) -> Optimizer:
        """Get the optimizer from the strategy at the particular iterations number

        Parameters
        ----------
        iteration_number
            Iteratation number

        Returns
        -------
            An optimizerS
        """
        ...


class BiOptimizer_Strategy(Strategy):
    def __init__(self, optimizer_1: Optimizer, optimizer_2: Optimizer):
        """Meta optimization strategy where we use one optimizer for half of the time and the other one for the second half

        Parameters
        ----------
        optimizer_1
            Optimizer to be used first half of the total number of iterations
        optimizer_2
            Optimizer to be used for the second half of the total number of iterations
        """
        self.optimizer_1 = optimizer_1
        self.optimizer_2 = optimizer_2

    def set_total_number_of_iterations(self, iterations: int):
        """Set the total number of iterations

        Parameters
        ----------
        iterations
            number of iterations in total
        """
        self.total_iterations = iterations

    def get_optimizer(self, iteration_progress: float) -> Optimizer:
        # If first half of total number of iterations, pick first optimizer
        if iteration_progress < 0.5:
            print(f"({iteration_progress}) selected optimizer_1")
            return self.optimizer_1

        else:
            print(f"({iteration_progress}) selected optimizer_2")
            return self.optimizer_2


class MetaOptimizer:
    def __init__(self, data: Data, strategy: Strategy):
        self.data = data
        self.strategy = strategy

    def update_step(self, function: Function, optimizer: Optimizer):
        optimizer.set_data(self.data)
        optimizer.iterate(iterations=optimizer.parameter.population, function=function)
        self.data = optimizer.extract_data()

    def iterate(self, iterations: int, function: Function):
        initial_samples = self.data.get_number_of_datapoints()
        while self.data.get_number_of_datapoints() < (iterations + initial_samples):
            self.update_step(
                function=function,
                optimizer=self.strategy.get_optimizer(
                    self.data.get_number_of_datapoints() / (iterations + initial_samples)
                ),
            )

        # Remove overiterations
        self.data.remove_rows_bottom(self.data.get_number_of_datapoints() - iterations + initial_samples)

    def extract_data(self) -> Data:
        """Returns a copy of the data

        Returns
        -------
            copy of the data
        """
        return copy(self.data)
