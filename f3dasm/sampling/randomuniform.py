import numpy as np

from ..src.samplingmethod import SamplingMethod


class RandomUniform(SamplingMethod):
    """Sampling via random uniform sampling"""
    def sample(self, numsamples: int, dimensions: int) -> np.array:

        if self.seed:
            np.random.seed(self.seed)

        samples = np.random.uniform(size=(numsamples, dimensions))
        return samples
