from abc import ABC
from dataclasses import dataclass
from typing import Any, List
import numpy as np
import pandas as pd

from .designofexperiments import DoE


@dataclass
class SamplingMethod(ABC):
    """Interface for sampling method

    Args:
        doe (DoE): design of experiments object
        seed (Any): Optional: seed for sampling (default is None)

    """

    doe: DoE
    seed: Any = None

    def __post_init__(self):
        if self.seed:
            np.random.seed(self.seed)

    def sample_continuous(self, numsamples: int, doe: DoE) -> np.ndarray:
        """Create N samples within the search space

        Args:
            numsamples (int): number of samples
            dimensions (int): number of dimensions

        Returns:
            np.array: samples
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def get_samples(self, numsamples: int) -> pd.DataFrame:
        """Receive samples of the search space"""
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

        # Get the column names in this particular order
        columnnames = [
            ("input", name)
            for name in self.doe.get_continuous_names()
            + self.doe.get_discrete_names()
            + self.doe.get_categorical_names()
        ]

        df = self.cast_to_dataframe(samples=samples, columnnames=columnnames)
        return df

    def cast_to_dataframe(
        self, samples: np.ndarray, columnnames: List[str]
    ) -> pd.DataFrame:
        """Cast the samples to a DataFrame"""

        # First get an empty reference frame from the DoE
        empty_frame = self.doe.get_empty_dataframe()

        # Then, create a new frame from the samples and columnnames
        samples_frame = pd.DataFrame(data=samples, columns=columnnames)

        # Concat the two frames
        df = pd.concat([empty_frame, samples_frame], sort=True)

        # Apparently you need to cast the types again
        df = df.astype(self.doe.cast_types_dataframe(self.doe.input_space, "input"))
        df = df.astype(self.doe.cast_types_dataframe(self.doe.output_space, "output"))

        return df

    def sample_discrete(self, numsamples: int, doe: DoE):
        """Sample the descrete parameters, default randomly uniform"""
        discrete = doe.get_discrete_parameters()
        samples = np.empty(shape=(numsamples, len(discrete)))
        for dim, _ in enumerate(discrete):
            samples[:, dim] = np.random.choice(
                range(discrete[dim].lower_bound, discrete[dim].upper_bound + 1),
                size=numsamples,
            )

        return samples

    def sample_categorical(self, numsamples: int, doe: DoE):
        """Sample the categorical parameters, default randomly uniform"""
        categorical = doe.get_categorical_parameters()
        samples = np.empty(shape=(numsamples, len(categorical)), dtype=object)
        for dim, _ in enumerate(categorical):
            samples[:, dim] = np.random.choice(
                categorical[dim].categories, size=numsamples
            )

        return samples

    def stretch_samples(self, doe: DoE, samples: np.ndarray) -> np.ndarray:
        """Stretch samples to their boundaries"""
        continuous = doe.get_continuous_parameters()
        for dim, _ in enumerate(continuous):
            samples[:, dim] = (
                samples[:, dim]
                * (continuous[dim].upper_bound - continuous[dim].lower_bound)
                + continuous[dim].lower_bound
            )

        return samples


if __name__ == "__main__":  # pragma: no cover
    pass
