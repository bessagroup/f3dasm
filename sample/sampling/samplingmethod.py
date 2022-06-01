from abc import ABC
from dataclasses import dataclass
import numpy as np

from designofexperiments import DesignOfExperiments


@dataclass
class SamplingMethod(ABC):
    """Interface for sampling method"""
    doe: DesignOfExperiments

    def sample(self, numsamples: int, dimensions: int) -> np.array:
        """Create N samples within the search space

        Args:
            numsamples (int): number of samples
            dimensions (int): number of dimensions

        Returns:
            np.array: samples
        """
        raise NotImplementedError
