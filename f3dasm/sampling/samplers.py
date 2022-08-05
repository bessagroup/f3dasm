import numpy as np
from SALib.sample import latin, sobol_sequence

from ..base.design import DesignSpace
from ..base.samplingmethod import SamplingInterface


class LatinHypercubeSampling(SamplingInterface):
    """Sampling via Latin Hypercube Sampling"""

    def sample_continuous(self, numsamples: int, designspace: DesignSpace) -> np.ndarray:
        continuous = designspace.get_continuous_parameters()
        problem = {
            "num_vars": len(continuous),
            "names": [s.name for s in continuous],
            "bounds": [[s.lower_bound, s.upper_bound] for s in continuous],
        }

        samples = latin.sample(problem, N=numsamples, seed=self.seed)
        return samples


class RandomUniformSampling(SamplingInterface):
    """Sampling via random uniform sampling"""

    def sample_continuous(self, numsamples: int, designspace: DesignSpace) -> np.ndarray:
        continuous = designspace.get_continuous_parameters()
        dimensions = len(continuous)

        samples = np.random.uniform(size=(numsamples, dimensions))

        # stretch samples
        samples = self.stretch_samples(designspace, samples)
        return samples


class SobolSequenceSampling(SamplingInterface):
    """Sampling via Sobol Sequencing with SALib"""

    def sample_continuous(self, numsamples: int, designspace: DesignSpace) -> np.ndarray:
        continuous = designspace.get_continuous_parameters()
        dimensions = len(continuous)

        samples = sobol_sequence.sample(numsamples, dimensions)

        # stretch samples
        samples = self.stretch_samples(designspace, samples)
        return samples
