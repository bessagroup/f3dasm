"""Base class for sampling methods"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
import sys
from itertools import product

if sys.version_info < (3, 8):  # NOQA
    from typing_extensions import Literal  # NOQA
else:
    from typing import Literal

from typing import Optional

# Third-party
import numpy as np
import pandas as pd
from SALib.sample import latin, sobol_sequence

# Locals
from .domain import Domain

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

SamplerNames = Literal['random', 'latin', 'sobol', 'grid']

#                                                              Factory function
# =============================================================================


def _sampler_factory(sampler: str, domain: Domain) -> Sampler:
    if sampler.lower() == 'random':
        return RandomUniform(domain)

    elif sampler.lower() == 'latin':
        return LatinHypercube(domain)

    elif sampler.lower() == 'sobol':
        return SobolSequence(domain)

    elif sampler.lower() == 'grid':
        return GridSampler(domain)

    else:
        raise KeyError(f"Sampler {sampler} not found!"
                       f"Available built-in samplers are: 'random',"
                       f"'latin' and 'sobol'")


#                                                                    Base Class
# =============================================================================


class Sampler:
    def __init__(self, domain: Domain, seed: Optional[int] = None,
                 number_of_samples: Optional[int] = None):
        """Interface for sampling method

        Parameters
        ----------
        domain : Domain
            domain object
        seed : int
            seed for sampling
        number_of_samples : Optional[int]
            number of samples to be generated, defaults to None
        """
        self.domain = domain
        self.seed = seed
        self.number_of_samples = number_of_samples
        if seed:
            np.random.seed(seed)

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

    def get_samples(self, numsamples: Optional[int] = None) -> pd.DataFrame:
        """Receive samples of the search space

        Parameters
        ----------
        numsamples
            number of samples

        Returns
        -------
            Data objects with the samples
        """

        self.set_seed(self.seed)

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
            (samples_continuous, samples_discrete,
             samples_categorical, samples_constant))

        # TODO #60 : Fix this ordering issue
        # Get the column names in this particular order
        columnnames = [
            name
            for name in self.domain.get_continuous_names(
            ) + self.domain.get_discrete_names(
            ) + self.domain.get_categorical_names(
            ) + self.domain.get_constant_names()
        ]

        # First get an empty reference frame from the DoE
        empty_frame = self.domain._create_empty_dataframe()

        # Then, create a new frame from the samples and columnnames
        samples_frame = pd.DataFrame(
            data=samples, columns=columnnames, dtype=object)
        df = pd.concat([empty_frame, samples_frame], sort=True)

        return df

    def __call__(self, domain: Domain, n_samples: int, seed: int):
        """Call the sampler"""
        self.domain = domain
        self.number_of_samples = n_samples
        self.seed = seed
        return self.get_samples()

    def _sample_constant(self, numsamples: int):
        constant = self.domain.get_constant_parameters()
        samples = np.empty(shape=(numsamples, len(constant)))
        for dim, param in enumerate(constant.values()):
            samples[:, dim] = param.value

        return samples

    def _sample_discrete(self, numsamples: int):
        """Sample the descrete parameters, default randomly uniform"""
        discrete = self.domain.get_discrete_parameters()
        samples = np.empty(shape=(numsamples, len(discrete)), dtype=np.int32)
        for dim, param in enumerate(discrete.values()):
            samples[:, dim] = np.random.choice(
                range(param.lower_bound,
                      param.upper_bound + 1, param.step),
                size=numsamples,
            )

        return samples

    def _sample_categorical(self, numsamples: int):
        """Sample the categorical parameters, default randomly uniform"""
        categorical = self.domain.get_categorical_parameters()
        samples = np.empty(shape=(numsamples, len(categorical)), dtype=object)
        for dim, param in enumerate(categorical.values()):
            samples[:, dim] = np.random.choice(
                param.categories, size=numsamples)

        return samples

    def _stretch_samples(self, samples: np.ndarray) -> np.ndarray:
        """Stretch samples to their boundaries"""
        continuous = self.domain.get_continuous_parameters()
        for dim, param in enumerate(continuous.values()):
            samples[:, dim] = (
                samples[:, dim] * (
                    param.upper_bound - param.lower_bound
                ) + param.lower_bound
            )

            # If param.log is True, take the 10** of the samples
            if param.log:
                samples[:, dim] = 10**samples[:, dim]

        return samples


#                                                             Built-in samplers
# =============================================================================

class LatinHypercube(Sampler):
    """Sampling via Latin Hypercube Sampling"""

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
        continuous = self.domain.continuous
        problem = {
            "num_vars": len(continuous),
            "names": continuous.names,
            "bounds": [[s.lower_bound, s.upper_bound]
                       for s in continuous.values()],
        }

        samples = latin.sample(problem, N=numsamples, seed=self.seed)
        return samples


class RandomUniform(Sampler):
    """
    Sampling via random uniform sampling
    """

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
        continuous = self.domain.continuous
        samples = np.random.uniform(size=(numsamples, len(continuous)))

        # stretch samples
        samples = self._stretch_samples(samples)
        return samples


class SobolSequence(Sampler):
    """Sampling via Sobol Sequencing with SALib

    Reference: `SALib <https://salib.readthedocs.io/en/latest/>`_"""

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
        continuous = self.domain.continuous

        samples = sobol_sequence.sample(numsamples, len(continuous))

        # stretch samples
        samples = self._stretch_samples(samples)
        return samples


class GridSampler(Sampler):
    """Sampling via Grid Sampling

    All the combination of the discrete and categorical parameters are
    sampled. The argument number_of_samples is ignored.
    Notes
    -----
    This sampler is at the moment only applicable for
    discrete and categorical parameters.

    """

    def get_samples(self, numsamples: Optional[int] = None) -> pd.DataFrame:
        """Receive samples of the search space

        Parameters
        ----------
        numsamples
            number of samples

        Returns
        -------
            Data objects with the samples
        """

        self.set_seed(self.seed)

        # If numsamples is None, take the object attribute number_of_samples
        if numsamples is None:
            numsamples = self.number_of_samples

        continuous = self.domain.get_continuous_parameters()

        if continuous:
            raise ValueError("Grid sampling is only possible for domains \
                             strictly with only discrete and \
                            categorical parameters")

        discrete = self.domain.get_discrete_parameters()
        categorical = self.domain.get_categorical_parameters()

        _iterdict = {}

        for k, v in categorical.items():
            _iterdict[k] = v.categories

        for k, v, in discrete.items():
            _iterdict[k] = range(v.lower_bound, v.upper_bound+1)

        return pd.DataFrame(list(product(*_iterdict.values())),
                            columns=_iterdict, dtype=object)
