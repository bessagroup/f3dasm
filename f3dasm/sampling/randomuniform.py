import numpy as np

from ..src.samplingmethod import SamplingMethod


class RandomUniform(SamplingMethod):
    """Sampling via random uniform sampling"""
    def sample(self, numsamples: int, dimensions: int) -> np.array:
        samples = np.random.uniform(size=(numsamples, dimensions))
        return samples
