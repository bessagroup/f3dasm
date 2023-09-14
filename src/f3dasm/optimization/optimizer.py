"""
Module containing the interface class Optimizer
"""

#                                                                       Modules
# =============================================================================

# Standard
import json
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Tuple

# Third-party core
import numpy as np

# Locals
from ..datageneration.datagenerator import DataGenerator
from ..datageneration.functions.function import Function
from ..design.domain import Domain
from ..design.experimentdata import ExperimentData
from ..design.experimentsample import ExperimentSample

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


@dataclass
class OptimizerParameters:
    """Interface of a continuous benchmark function

    Parameters
    ----------
    population : int
        population of the optimizer update step
    force_bounds : bool
        force the optimizer to not exceed the boundaries of the domain
    """

    population: int = 1
    force_bounds: bool = True


class Optimizer:
    type: ClassVar[str] = 'any'
    hyperparameters: OptimizerParameters = OptimizerParameters()

    def __init__(self, domain: Domain, hyperparameters: Optional[Dict[str, Any]] = None,
                 seed: Optional[int] = None, name: Optional[str] = None):
        """Optimizer class for the optimization of a data-driven process

        Parameters
        ----------
        domain : Domain
            Domain indicating the search-space of the optimization parameters
        hyperparameters : Optional[Dict[str, Any]], optional
            Hyperparameters of the optimizer, by default None, it will use the default hyperparameters
        seed : Optional[int], optional
            Seed of the random number generator for stochastic optimization processes, by default None, set to random
        name : Optional[str], optional
            Name of the optimization object, by default None, it will use the name of the class
        """
        # Create an empty dictionary when hyperparameters is None
        if hyperparameters is None:
            hyperparameters = {}

        # Overwrite the default hyperparameters with the given hyperparameters
        self.hyperparameters.__init__(**hyperparameters)

        # Set the name of the optimizer to the class name if no name is given
        if name is None:
            name = self.__class__.__name__

        # Set the seed to a random number if no seed is given
        if seed is None:
            seed = np.random.randint(low=0, high=1e5)

        self.domain = domain
        self.seed = seed
        self.name = name
        self.__post_init__()

    def __post_init__(self):
        self._check_imports()
        self.set_seed()
        self.init_data()
        self.set_algorithm()

    @staticmethod
    def _check_imports():
        ...

    def init_data(self):
        """Set the data atrribute to an empty ExperimentData object"""
        self.data = ExperimentData(self.domain)

    def set_algorithm(self):
        """Set the algorithm attribute to the algorithm of choice"""
        ...

    def _construct_model(self, data_generator: DataGenerator):
        ...

    def _check_number_of_datapoints(self):
        """Check if the number of datapoints is sufficient for the initial population

        Raises
        ------
        ValueError
            Raises then the number of datapoints is insufficient
        """
        if len(self.data) < self.hyperparameters.population:
            raise ValueError(
                f'There are {len(self.data)} datapoints available, \
                     need {self.hyperparameters.population} for initial population!'
            )

    def set_seed(self):
        """Set the seed of the random number generator"""
        ...

    def reset(self):
        """Reset the optimizer to its initial state"""
        self.__post_init__()

    def set_data(self, data: ExperimentData):
        """Set the data attribute to the given data"""
        self.data = data

    def set_x0(self, experiment_data: ExperimentData):
        """Set the initial population to the best n samples of the given data

        Parameters
        ----------
        experiment_data : ExperimentData
            Data to be used for the initial population

        """
        x0 = experiment_data.get_n_best_output(self.hyperparameters.population)
        x0.reset_index()
        self.data = x0

    def get_name(self) -> str:
        """Get the name of the optimizer

        Returns
        -------
        str
            name of the optimizer
        """
        return self.name

    def get_info(self) -> List[str]:
        """Give a list of characteristic features of this optimizer

        Returns
        -------
            List of strings denoting the characteristics of this optimizer
        """
        return []

    def update_step(self, data_generator: DataGenerator) -> ExperimentData:
        """Update step of the optimizer. Needs to be implemented by the child class

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
        """
        raise NotImplementedError(
            "You should implement an update step for your algorithm!")
