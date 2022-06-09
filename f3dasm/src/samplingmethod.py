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

    def sample(self, numsamples: int, doe: DoE) -> np.array:
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
        continuous = DoE.getContinuousParameters()
        samples_unstretched = self.sample(
            numsamples=numsamples, dimensions=len(continuous)
        )

        # Stretch the samples according to their box-constraint boundaries
        samples = self.stretch_samples(samples_unstretched)

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
