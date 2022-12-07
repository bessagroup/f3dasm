#                                                                       Modules
# =============================================================================

# Standard
from abc import ABC
from copy import copy
from typing import List, Tuple

# Locals
from ..base.data import Data
from ..base.function import Function
from ..base.optimization import Optimizer, OptimizerParameters

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class OptimizationOrder:
    def __init__(self, number_of_iterations: int, optimizer: Optimizer):
        """Order for the optimizer to execute

        Parameters
        ----------
        number_of_iterations
            number of points to iterate
        optimizer
            optimizer that determines the update step
        """
        self.number_of_iterations = number_of_iterations
        self.optimizer = optimizer


class Strategy(ABC):
    def create_strategy(self, iterations: int) -> Tuple[OptimizationOrder]:
        ...


class EqualParts_Strategy(Strategy):
    def __init__(self, optimizers: List[Optimizer]):
        """Meta optimization strategy where we use one optimizer for half of the time 
        and the other one for the second half

        Parameters
        ----------
        optimizers
            List of optimizers in order
        """
        self.optimizers = optimizers

    def create_strategy(self, iterations: int) -> Tuple[OptimizationOrder]:
        strategy: Tuple[OptimizationOrder] = tuple(
            OptimizationOrder(number_of_iterations=iterations //
                              len(self.optimizers), optimizer=optimizer)
            for optimizer in self.optimizers
        )
        # Add the remained of the iterations to the last part
        strategy[-1].number_of_iterations += iterations % len(self.optimizers)

        return strategy


class MetaOptimizer(Optimizer):
    def __init__(self, data: Data, strategy: Strategy, seed: int):
        """Meta optimizer class: executing multiple optimizers during training

        Parameters
        ----------
        data
            data object
        strategy
            optimization strategy
        seed
            seed for random number generator
        """
        self.strategy = strategy
        self.parameter = OptimizerParameters()
        super().__init__(data=data, seed=seed)

    def update_step(self, iterations: int, function: Function, optimizer: Optimizer):
        optimizer.set_data(self.data)
        optimizer.iterate(iterations=iterations, function=function)
        self.data = optimizer.extract_data()

    def iterate(self, iterations: int, function: Function):
        number_of_initial_samples = self.data.get_number_of_datapoints()
        optimization_order: Tuple[OptimizationOrder] = self.strategy.create_strategy(
            iterations)

        for order in optimization_order:
            self.update_step(iterations=order.number_of_iterations,
                             function=function, optimizer=order.optimizer)

        # Remove overiterations
        self.data.remove_rows_bottom(
            self.data.get_number_of_datapoints() - iterations - number_of_initial_samples)

    def get_name(self) -> str:
        """Retrieve the name of the optimizers

        Returns
        -------
            name of the optimizer
        """
        return f"{self.__class__.__name__}_{self.strategy.__class__.__name__}"
