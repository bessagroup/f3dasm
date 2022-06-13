from abc import ABC
from dataclasses import dataclass
from typing import Any
import numpy as np

from .designofexperiments import DoE


@dataclass
class SamplingMethod(ABC):
    """Interface for sampling method"""

    doe: DoE
    seed: Any = None

    def __post_init__(self):
        if self.seed:
            np.random.seed(self.seed)

    def sample_continuous(self, numsamples: int, doe: DoE) -> np.array:
        """Create N samples within the search space

        Args:
            numsamples (int): number of samples
            dimensions (int): number of dimensions

        Returns:
            np.array: samples
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def get_samples(self, numsamples: int):
        # First sample the continuous parameters
        samples_continuous = self.sample_continuous(numsamples=numsamples, doe=self.doe)

        # Sample discrete parameters
        samples_discrete = self.sample_discrete(numsamples=numsamples, doe=self.doe)

        # Sample categorical parameters
        samples_categorical = self.sample_categorical(
            numsamples=numsamples, doe=self.doe
        )
        # Merge samples into array
        samples = np.hstack((samples_continuous, samples_discrete, samples_categorical))
        return samples

    def sample_discrete(self, numsamples: int, doe: DoE):
        discrete = doe.getDiscreteParameters()
        samples = np.empty(shape=(numsamples, len(discrete)))
        for dim, _ in enumerate(discrete):
            samples[:, dim] = np.random.choice(
                range(discrete[dim].lower_bound, discrete[dim].upper_bound + 1),
                size=numsamples,
            )

        return samples

    def sample_categorical(self, numsamples: int, doe: DoE):
        categorical = doe.getCategoricalParameters()
        samples = np.empty(shape=(numsamples, len(categorical)), dtype=object)
        for dim, _ in enumerate(categorical):
            samples[:, dim] = np.random.choice(
                categorical[dim].categories, size=numsamples
            )

        return samples

    def stretch_samples(self, doe: DoE, samples: np.array):
        """Stretch samples"""
        continuous = doe.getContinuousParameters()
        for dim, _ in enumerate(continuous):
            samples[:, dim] = (
                samples[:, dim]
                * (continuous[dim].upper_bound - continuous[dim].lower_bound)
                + continuous[dim].lower_bound
            )

        return samples
