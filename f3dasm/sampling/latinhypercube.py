from typing import Any
import numpy as np
from SALib.sample import latin

from ..src.samplingmethod import SamplingMethod


class LatinHypercube(SamplingMethod):
    """Sampling via Latin Hypercube Sampling"""
    def sample(self, numsamples: int, dimensions: int, seed: Any | None = None) -> np.array:
        problem = {
            'num_vars': dimensions,
            'names': ['x' + str(n) for n in range(dimensions)],
            'bounds': [[0., 1.] for n in range(dimensions)]
        }

        samples = latin.sample(problem, numsamples, seed=seed)
        return samples
