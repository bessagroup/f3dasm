import numpy as np
from SALib.sample import sobol_sequence

from f3dasm.base.designofexperiments import DesignSpace

from ..base.samplingmethod import SamplingInterface


class SobolSequenceSampling(SamplingInterface):
    """Sampling via Sobol Sequencing with SALib"""

    def sample_continuous(self, numsamples: int, doe: DesignSpace) -> np.ndarray:
        continuous = doe.get_continuous_parameters()
        dimensions = len(continuous)

        samples = sobol_sequence.sample(numsamples, dimensions)

        # stretch samples
        samples = self.stretch_samples(doe, samples)
        return samples
