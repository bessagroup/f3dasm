"""Base class for sampling methods"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
import json
from typing import Any, List, Optional

# Third-party core
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig

# Locals
from ..design.domain import Domain
from ..design.experimentdata import ExperimentData

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class Sampler:
    def __init__(self, design: Domain, seed: Optional[int] = None, number_of_samples: Optional[int] = None):
        """Interface for sampling method

        Parameters
        ----------
        design : Domain
            design of experiments object
        seed : int
            seed for sampling
        number_of_samples : Optional[int]
            number of samples to be generated, defaults to None
        """
        self.design = design
        self.seed = seed
        self.number_of_samples = number_of_samples
        if seed:
            np.random.seed(seed)

    @classmethod
    def from_yaml(cls, domain_config: DictConfig, sampler_config: DictConfig) -> Sampler:
        """Create a sampler from a yaml configuration"""

        args = {**sampler_config, 'design': None}
        sampler: Sampler = instantiate(args)
        sampler.design = Domain.from_yaml(domain_config)
        return sampler

    def set_seed(self, seed: int):
        """Set the seed of the sampler

        Parameters
        ----------
        seed
            the seed to be used
        """
        np.random.seed(seed)
        self.seed = seed

    def sample_continuous(self, numsamples: int) -> np.ndarray:
        """Create N samples within the search space

        Parameters
        ----------
        numsamples
            number of samples

        Returns
        -------
            samples
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def get_samples(self, numsamples: Optional[int] = None) -> ExperimentData:
        """Receive samples of the search space

        Parameters
        ----------
        numsamples
            number of samples

        Returns
        -------
            Data objects with the samples
        """
        # If numsamples is None, take the object attribute number_of_samples
        if numsamples is None:
            numsamples = self.number_of_samples

        # First sample the continuous parameters
        samples_continuous = self.sample_continuous(numsamples=numsamples)

        # Sample discrete parameters
        samples_discrete = self._sample_discrete(numsamples=numsamples)

        # Sample categorical parameters
        samples_categorical = self._sample_categorical(numsamples=numsamples)

        # Sample constant parameters
        samples_constant = self._sample_constant(numsamples=numsamples)

        # Merge samples into array
        samples = np.hstack(
            (samples_continuous, samples_discrete, samples_categorical, samples_constant))

        # TODO #60 : Fix this ordering issue
        # Get the column names in this particular order
        columnnames = [
            name
            for name in self.design.get_continuous_names(
            ) + self.design.get_discrete_names(
            ) + self.design.get_categorical_names(
            ) + self.design.get_constant_names()
        ]

        data = self._cast_to_data_object(
            samples=samples, columnnames=columnnames)
        return data

    def _cast_to_data_object(self, samples: np.ndarray, columnnames: List[str]) -> ExperimentData:
        """Cast the samples to a Data object"""
        data = ExperimentData(domain=self.design)

        # First get an empty reference frame from the DoE
        empty_frame = self.design._create_empty_dataframe()

        # Then, create a new frame from the samples and columnnames
        samples_frame = pd.DataFrame(data=samples, columns=columnnames)
        df = pd.concat([empty_frame, samples_frame], sort=True)

        # Add the samples to the Data object
        data.add(data=df)

        return data

    def _sample_constant(self, numsamples: int):
        constant = self.design.get_constant_parameters()
        samples = np.empty(shape=(numsamples, len(constant)))
        for dim, param in enumerate(constant.values()):
            samples[:, dim] = param.value

        return samples

    def _sample_discrete(self, numsamples: int):
        """Sample the descrete parameters, default randomly uniform"""
        discrete = self.design.get_discrete_parameters()
        samples = np.empty(shape=(numsamples, len(discrete)))
        for dim, param in enumerate(discrete.values()):
            samples[:, dim] = np.random.choice(
                range(param.lower_bound,
                      param.upper_bound + 1),
                size=numsamples,
            )

        return samples

    def _sample_categorical(self, numsamples: int):
        """Sample the categorical parameters, default randomly uniform"""
        categorical = self.design.get_categorical_parameters()
        samples = np.empty(shape=(numsamples, len(categorical)), dtype=object)
        for dim, param in enumerate(categorical.values()):
            samples[:, dim] = np.random.choice(
                param.categories, size=numsamples)

        return samples

    def _stretch_samples(self, samples: np.ndarray) -> np.ndarray:
        """Stretch samples to their boundaries"""
        continuous = self.design.get_continuous_parameters()
        for dim, param in enumerate(continuous.values()):
            samples[:, dim] = (
                samples[:, dim] * (
                    param.upper_bound - param.lower_bound
                ) + param.lower_bound
            )

        return samples
