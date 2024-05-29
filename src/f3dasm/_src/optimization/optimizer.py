"""
Module containing the interface class Optimizer
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
from typing import ClassVar, Iterable, List, Protocol, Tuple

# Third-party core
import numpy as np
import pandas as pd

# Locals
from ..datageneration.datagenerator import DataGenerator

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class ExperimentData(Protocol):
    @property
    def index(self, index) -> pd.Index:
        ...

    def get_n_best_output(self, n_samples: int) -> ExperimentData:
        ...

    def to_numpy() -> Tuple[np.ndarray, np.ndarray]:
        ...

    def select(self, indices: int | slice | Iterable[int]) -> ExperimentData:
        ...

# =============================================================================


class Optimizer:
    """
    Abstract class for optimization algorithms
    To implement a new optimizer, inherit from this class and implement the
    update_step method.

    Note
    ----
    The update_step method should have the following signature:

    '''
    def update_step(self, data_generator: DataGenerator)
    -> Tuple[np.ndarray, np.ndarray]:
    '''

    The method should return a tuple containing the new samples and the
    corresponding objective values. The new samples should be a numpy array.

    If the optimizer requires gradients, set the require_gradients attribute to
    True. This will ensure that the data_generator will calculate the gradients
    of the objective function from the DataGenerator.dfdx method.

    Hyperparameters can be set in the __init__ method of the child class.
    There are two hyperparameters that have special meaning:
    - population: the number of individuals in the population
    - seed: the seed of the random number generator

    You can create extra methods in your child class as you please, however
    it is advised not to create private methods (methods starting with an
    underscore) as these might be used in the base class.
    """
    type: ClassVar[str] = 'any'
    require_gradients: ClassVar[bool] = False

#                                                            Private Properties
# =============================================================================

    @property
    def _seed(self) -> int:
        """
        Property to return the seed of the optimizer

        Returns
        -------
        int | None
            Seed of the optimizer

        Note
        ----
        If the seed is not set, the property will return None
        This is done to prevent errors when the seed is not an available
        attribute in a custom optimizer class.
        """
        return self.seed if hasattr(self, 'seed') else None

    @property
    def _population(self) -> int:
        """
        Property to return the population size of the optimizer

        Returns
        -------
        int
            Number of individuals in the population

        Note
        ----
        If the population is not set, the property will return 1
        This is done to prevent errors when the population size is not an
        available attribute in a custom optimizer class.
        """
        return self.population if hasattr(self, 'population') else 1

#                                                                Public Methods
# =============================================================================

    def update_step(self, data_generator: DataGenerator) -> ExperimentData:
        """Update step of the optimizer. Needs to be implemented
         by the child class

        Parameters
        ----------
        data_generator : DataGenerator
            data generator object to calculate the objective value

        Returns
        -------
        ExperimentData
            ExperimentData object containing the new samples

        Raises
        ------
        NotImplementedError
            Raises when the method is not implemented by the child class

        Note
        ----
        You can access the data attribute of the optimizer to get the
        available data points. The data attribute is an
        f3dasm.ExperimentData object.
        """
        raise NotImplementedError(
            "You should implement an update step for your algorithm!")

#                                                               Private Methods
# =============================================================================

    def _set_algorithm(self):
        """
        Method that can be implemented to set the optimization algorithm.
        Whenever the reset method is called, this method will be called to
        reset the algorithm to its initial state."""
        ...

    def _construct_model(self, data_generator: DataGenerator):
        """
        Method that is called before the optimization starts. This method can
        be used to construct a model based on the available data or a specific
        data generator.

        Parameters
        ----------
        data_generator : DataGenerator
            DataGenerator object

        Note
        ----
        When this method is not implemented, the method will do nothing.
        """
        ...

    def _check_number_of_datapoints(self):
        """
        Check if the number of datapoints is sufficient for the
        initial population

        Raises
        ------
        ValueError
            Raises when the number of datapoints is insufficient
        """
        if len(self.data) < self._population:
            raise ValueError(
                f'There are {len(self.data)} datapoints available, \
                     need {self._population} for initial \
                         population!'
            )

    def _reset(self, data: ExperimentData):
        """Reset the optimizer to its initial state

        Parameters
        ----------
        data : ExperimentData
            Data to set the optimizer to its initial state

        Note
        ----
        This method should be called whenever the optimizer is reset
        to its initial state. This can be done when the optimizer is
        re-initialized or when the optimizer is re-used for a new
        optimization problem.

        The following steps are taken when the reset method is called:
        - The data attribute is set to the given data (self._set_data)
        - The algorithm is set to its initial state (self._set_algorithm)
        """
        self._set_data(data)
        self._set_algorithm()

    def _set_data(self, data: ExperimentData):
        """Set the data attribute to the given data

        Parameters
        ----------
        data : ExperimentData
            Data to set the optimizer to its initial state
        """
        self.data = data

    def _get_info(self) -> List[str]:
        """Give a list of characteristic features of this optimizer

        Returns
        -------
        List[str]
            List of characteristics of the optimizer
        """
        return []
