import numpy as np
from SALib.sample import sobol_sequence

from f3dasm.src.designofexperiments import DoE

from ..src.samplingmethod import SamplingMethod


class SobolSequencing(SamplingMethod):
    """Sampling via Sobol Sequencing with SALib"""

    def sample(self, numsamples: int, doe: DoE) -> np.array:
        continuous = doe.getContinuousParameters()
        dimensions = len(continuous)

        samples = sobol_sequence.sample(numsamples, dimensions)

        # stretch samples
        samples = self.stretch_samples(doe, samples)
        return samples
