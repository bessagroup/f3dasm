#                                                                       Modules
# =============================================================================

# Standard
from abc import ABC
from dataclasses import dataclass
from typing import Any, List

# Third-party
import autograd.numpy as np
import pandas as pd

# Locals
from ..base.data import Data
from ..base.design import DesignSpace

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


@dataclass
class SamplingInterface(ABC):
    """Interface for sampling method

    Parameters
    ----------
    design
        design of experiments object
    seed
        seed for sampling
    """

    design: DesignSpace
    seed: Any or int = None

    def __post_init__(self):
        if self.seed:
            np.random.seed(self.seed)

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

        :param numsamples: number of samples
        :returns: samples
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def get_samples(self, numsamples: int) -> Data:
        """Receive samples of the search space

        Parameters
        ----------
        numsamples
            number of samples

        Returns
        -------
            Data objects with the samples
        """
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
            ("input", name)
            for name in self.design.get_continuous_input_names()
            + self.design.get_discrete_input_names()
            + self.design.get_categorical_input_names()
            + self.design.get_constant_input_names()
        ]

        data = self._cast_to_data_object(
            samples=samples, columnnames=columnnames)
        return data

    def _cast_to_data_object(self, samples: np.ndarray, columnnames: List[str]) -> Data:
        """Cast the samples to a Data object"""
        data = Data(design=self.design)

        # First get an empty reference frame from the DoE
        empty_frame = self.design.get_empty_dataframe()

        # Then, create a new frame from the samples and columnnames
        samples_frame = pd.DataFrame(data=samples, columns=columnnames)
        df = pd.concat([empty_frame, samples_frame], sort=True)

        # Add the samples to the Data object
        data.add(data=df)

        return data

    def _sample_constant(self, numsamples: int):
        constant = self.design.get_constant_input_parameters()
        samples = np.empty(shape=(numsamples, len(constant)))
        for dim, _ in enumerate(constant):
            samples[:, dim] = constant[dim].value

        return samples

    def _sample_discrete(self, numsamples: int):
        """Sample the descrete parameters, default randomly uniform"""
        discrete = self.design.get_discrete_input_parameters()
        samples = np.empty(shape=(numsamples, len(discrete)))
        for dim, _ in enumerate(discrete):
            samples[:, dim] = np.random.choice(
                range(discrete[dim].lower_bound,
                      discrete[dim].upper_bound + 1),
                size=numsamples,
            )

        return samples

    def _sample_categorical(self, numsamples: int):
        """Sample the categorical parameters, default randomly uniform"""
        categorical = self.design.get_categorical_input_parameters()
        samples = np.empty(shape=(numsamples, len(categorical)), dtype=object)
        for dim, _ in enumerate(categorical):
            samples[:, dim] = np.random.choice(
                categorical[dim].categories, size=numsamples)

        return samples

    def _stretch_samples(self, samples: np.ndarray) -> np.ndarray:
        """Stretch samples to their boundaries"""
        continuous = self.design.get_continuous_input_parameters()
        for dim, _ in enumerate(continuous):
            samples[:, dim] = (
                samples[:, dim] * (continuous[dim].upper_bound -
                                   continuous[dim].lower_bound)
                + continuous[dim].lower_bound
            )

        return samples
