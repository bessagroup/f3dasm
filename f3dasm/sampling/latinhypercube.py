import autograd.numpy as np
from SALib.sample import latin

from ..base.samplingmethod import SamplingInterface


class LatinHypercube(SamplingInterface):
    """Sampling via Latin Hypercube Sampling"""

    def sample_continuous(self, numsamples: int) -> np.ndarray:
        continuous = self.design.get_continuous_input_parameters()
        problem = {
            "num_vars": len(continuous),
            "names": [s.name for s in continuous],
            "bounds": [[s.lower_bound, s.upper_bound] for s in continuous],
        }

        samples = latin.sample(problem, N=numsamples, seed=self.seed)
        return samples
