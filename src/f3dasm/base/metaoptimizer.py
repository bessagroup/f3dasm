from abc import ABC
from copy import copy
from dataclasses import dataclass
from typing import List, Tuple

from ..base.data import Data
from ..base.function import Function
from ..base.optimization import Optimizer


@dataclass
class OptimizationOrder:
    number_of_iterations: int
    optimizer: Optimizer


class Strategy(ABC):
    def create_strategy(self, iterations: int) -> Tuple[OptimizationOrder]:
        ...


class EqualParts_Strategy(Strategy):
    def __init__(self, optimizers: List[Optimizer]):
        """Meta optimization strategy where we use one optimizer for half of the time and the other one for the second half

        Parameters
        ----------
        optimizer_1
            Optimizer to be used first half of the total number of iterations
        optimizer_2
            Optimizer to be used for the second half of the total number of iterations
        """
        self.optimizers = optimizers

    def create_strategy(self, iterations: int) -> Tuple[OptimizationOrder]:
        strategy: Tuple[OptimizationOrder] = tuple(
            OptimizationOrder(number_of_iterations=iterations // len(self.optimizers), optimizer=optimizer)
            for optimizer in self.optimizers
        )
        # Add the remained of the iterations to the last part
        strategy[-1].number_of_iterations += iterations % len(self.optimizers)

        return strategy


class MetaOptimizer:
    def __init__(self, data: Data, strategy: Strategy):
        self.data = data
        self.strategy = strategy

    def update_step(self, iterations: int, function: Function, optimizer: Optimizer):
        optimizer.set_data(self.data)
        optimizer.iterate(iterations=iterations, function=function)
        self.data = optimizer.extract_data()

    def iterate(self, iterations: int, function: Function):
        number_of_initial_samples = self.data.get_number_of_datapoints()
        optimization_order: Tuple[OptimizationOrder] = self.strategy.create_strategy(iterations)

        for order in optimization_order:
            self.update_step(iterations=order.number_of_iterations, function=function, optimizer=order.optimizer)

        # Remove overiterations
        self.data.remove_rows_bottom(self.data.get_number_of_datapoints() - iterations - number_of_initial_samples)

    def extract_data(self) -> Data:
        """Returns a copy of the data

        Returns
        -------
            copy of the data
        """
        return copy(self.data)
