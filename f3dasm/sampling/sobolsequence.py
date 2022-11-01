import autograd.numpy as np
from SALib.sample import sobol_sequence

from ..base.samplingmethod import SamplingInterface


class SobolSequence(SamplingInterface):
    """Sampling via Sobol Sequencing with SALib"""

    def sample_continuous(self, numsamples: int) -> np.ndarray:
        continuous = self.design.get_continuous_input_parameters()
        dimensions = len(continuous)

        samples = sobol_sequence.sample(numsamples, dimensions)

        # stretch samples
        samples = self._stretch_samples(samples)
        return samples
