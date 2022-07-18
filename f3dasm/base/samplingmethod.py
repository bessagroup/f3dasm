from abc import ABC
from dataclasses import dataclass
from typing import Any, List
import numpy as np
import pandas as pd

from f3dasm.base.data import Data

from .designofexperiments import DesignSpace


@dataclass
class SamplingInterface(ABC):
    """Interface for sampling method

    Args:
        doe (DoE): design of experiments object
        seed (Any): Optional: seed for sampling (default is None)

    """

    doe: DesignSpace
    seed: Any = None

    def __post_init__(self):
        if self.seed:
            np.random.seed(self.seed)

    def set_seed(self, seed: int) -> None:
        np.random.seed(seed)
        self.seed = seed

    def sample_continuous(
        self, numsamples: int, designspace: DesignSpace
    ) -> np.ndarray:
        """Create N samples within the search space

        Args:
            numsamples (int): number of samples
            dimensions (int): number of dimensions

        Returns:
            np.array: samples
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def get_samples(self, numsamples: int) -> Data:
        """Receive samples of the search space

        Args:
            numsamples (int): number of samples

        Returns:
            Data: Data objects with the samples
        """
        # First sample the continuous parameters
        samples_continuous = self.sample_continuous(
            numsamples=numsamples, designspace=self.doe
        )

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

        # df = self.cast_to_dataframe(samples=samples, columnnames=columnnames)
        data = self.cast_to_data_object(samples=samples, columnnames=columnnames)
        return data

    def cast_to_data_object(self, samples: np.ndarray, columnnames: List[str]) -> Data:
        """Cast the samples to a Data object"""
        data = Data(designspace=self.doe)

        # First get an empty reference frame from the DoE
        empty_frame = self.doe.get_empty_dataframe()

        # Then, create a new frame from the samples and columnnames
        samples_frame = pd.DataFrame(data=samples, columns=columnnames)
        df = pd.concat([empty_frame, samples_frame], sort=True)

        # Add the samples to the Data object
        data.add(data=df)

        return data

    def sample_discrete(self, numsamples: int, doe: DesignSpace):
        """Sample the descrete parameters, default randomly uniform"""
        discrete = doe.get_discrete_parameters()
        samples = np.empty(shape=(numsamples, len(discrete)))
        for dim, _ in enumerate(discrete):
            samples[:, dim] = np.random.choice(
                range(discrete[dim].lower_bound, discrete[dim].upper_bound + 1),
                size=numsamples,
            )

        return samples

    def sample_categorical(self, numsamples: int, doe: DesignSpace):
        """Sample the categorical parameters, default randomly uniform"""
        categorical = doe.get_categorical_parameters()
        samples = np.empty(shape=(numsamples, len(categorical)), dtype=object)
        for dim, _ in enumerate(categorical):
            samples[:, dim] = np.random.choice(
                categorical[dim].categories, size=numsamples
            )

        return samples

    def stretch_samples(self, doe: DesignSpace, samples: np.ndarray) -> np.ndarray:
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
