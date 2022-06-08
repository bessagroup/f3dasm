from abc import ABC
from dataclasses import dataclass
from typing import Any
import numpy as np

from .designofexperiments import DesignOfExperiments


@dataclass
class SamplingMethod(ABC):
    """Interface for sampling method"""

    doe: DesignOfExperiments
    seed: Any = None

    def __post_init__(self):
        if self.seed:
            np.random.seed(self.seed)

    def sample(self, numsamples: int, dimensions: int) -> np.array:
        """Create N samples within the search space

        Args:
            numsamples (int): number of samples
            dimensions (int): number of dimensions

        Returns:
            np.array: samples
        """
        raise NotImplementedError("Subclasses should implement this method.")
