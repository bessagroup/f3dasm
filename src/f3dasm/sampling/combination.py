#                                                                       Modules
# =============================================================================

from itertools import product
# Standard
from typing import Any

# Third-party
import numpy as np

# Locals
from ..design.domain import Domain
from .sampler import Sampler

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Work in progress'
# =============================================================================
#
# =============================================================================


class Combination(Sampler):
    """Sampling via all possible combinations"""

    def _sample_discrete(self, numsamples: int):
        """Sample the discrete parameters"""
        discrete = self.design.get_discrete_parameters()

        all_values = [set(range(param.lower_bound, param.upper_bound + 1))
                      for _, param in enumerate(discrete.values())]

        # Multiply all_values by the number of categorical samples
        n_categorical = 1
        categorical = self.design.get_categorical_parameters()
        for _, param in enumerate(categorical.values()):
            n_categorical *= len(param.categories)

        if n_categorical == 1:
            samples = np.array(list(product(*all_values)))

        else:
            _all_values = []
            a = list(product(*all_values))
            for i in range(n_categorical-1):
                _all_values.extend(a)
            samples = np.array(_all_values)

        return samples

    def _sample_categorical(self, numsamples: int):
        """Sample the categorical parameters"""
        categorical = self.design.get_categorical_parameters()

        all_values = [set(param.categories)
                      for dim, param in enumerate(categorical.values())]

        samples = np.array(list(product(*all_values)))

        # Multiply all_values by the number of discrete samples
        n_discrete = 1
        discrete = self.design.get_discrete_parameters()
        for _, param in enumerate(discrete.values()):
            n_discrete *= len(list(range(param.lower_bound, param.upper_bound)))

        if n_discrete == 1:
            samples = np.array(list(product(*all_values)))

        else:
            _all_values = []
            a = list(product(*all_values))
            for i in range(n_discrete-1):
                _all_values.extend(a)
            samples = np.array(_all_values)

        return samples

    def sample_continuous(self, numsamples: int) -> np.ndarray:
        """Sample from continuous space

        Parameters
        ----------
        numsamples
            number of samples

        Returns
        -------
            samples
        """

        # Multiply all_values by the number of categorical samples
        n_categorical = 0
        categorical = self.design.get_categorical_parameters()
        discrete = self.design.get_discrete_parameters()
        for _, param in enumerate(categorical.values()):
            n_categorical *= len(param.categories)

        for _, param in enumerate(discrete.values()):
            n_categorical *= (param.upper_bound - param.lower_bound + 1)

        continuous = self.design.get_continuous_parameters()
        dimensions = len(continuous)

        samples = np.random.uniform(size=(n_categorical, dimensions))

        # stretch samples
        samples = self._stretch_samples(samples)
        return samples
