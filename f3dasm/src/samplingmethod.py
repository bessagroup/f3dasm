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
        columnnames = [
            ["input"] * self.doe.getNumberOfParameters(),
            self.doe.getContinuousNames()
            + self.doe.getDiscreteNames()
            + self.doe.getCategoricalNames(),
        ]

        df = self.cast_to_dataframe(samples=samples, columnnames=columnnames)
        return df

    def cast_to_dataframe(
        self, samples: np.ndarray, columnnames: List[str]
    ) -> pd.DataFrame:
        """Cast the samples to a DataFrame"""
        # Make the dataframe
        df = pd.DataFrame(data=samples, columns=columnnames)

        # Make a dictionary that provides the datatype of each parameter
        coltypes = {}
        for continuous in self.doe.getContinuousNames():
            coltypes[("input", continuous)] = "float"
        for discrete in self.doe.getDiscreteNames():
            coltypes[("input", discrete)] = "int"
        for categorical in self.doe.getCategoricalNames():
            coltypes[("input", categorical)] = "category"

        # Cast the columns
        df = df.astype(coltypes)
        return df

    def sample_discrete(self, numsamples: int, doe: DoE):
        """Sample the descrete parameters, default randomly uniform"""
        discrete = doe.getDiscreteParameters()
        samples = np.empty(shape=(numsamples, len(discrete)))
        for dim, _ in enumerate(discrete):
            samples[:, dim] = np.random.choice(
                range(discrete[dim].lower_bound, discrete[dim].upper_bound + 1),
                size=numsamples,
            )

        return samples

    def sample_categorical(self, numsamples: int, doe: DoE):
        """Sample the categorical parameters, default randomly uniform"""
        categorical = doe.getCategoricalParameters()
        samples = np.empty(shape=(numsamples, len(categorical)), dtype=object)
        for dim, _ in enumerate(categorical):
            samples[:, dim] = np.random.choice(
                categorical[dim].categories, size=numsamples
            )

        return samples

    def stretch_samples(self, doe: DoE, samples: np.ndarray) -> np.ndarray:
        """Stretch samples to their boundaries"""
        continuous = doe.getContinuousParameters()
        for dim, _ in enumerate(continuous):
            samples[:, dim] = (
                samples[:, dim]
                * (continuous[dim].upper_bound - continuous[dim].lower_bound)
                + continuous[dim].lower_bound
            )

        return samples
