"""
Module containing the interface class Optimizer
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
from typing import ClassVar, Iterable, Protocol, Tuple

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
    def update_step(self)
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

    def update_step(self) -> ExperimentData:
        """Update step of the optimizer. Needs to be implemented
         by the child class

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
            "You should implement an update step for your optimizer!")

    def init(self, data: ExperimentData, data_generator: DataGenerator):
        """
        Initialize the optimizer with the given data and data generator

        Parameters
        ----------
        data : ExperimentData
            Data to initialize the optimizer
        data_generator : DataGenerator
            Data generator to calculate the objective value
        """
        raise NotImplementedError(
            "You should implement this method for your optimizer!")
