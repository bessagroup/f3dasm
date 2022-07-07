import numpy as np
from SALib.sample import latin

from f3dasm.base.designofexperiments import DesignSpace

from f3dasm.base.samplingmethod import SamplingInterface


class LatinHypercubeSampling(SamplingInterface):
    """Sampling via Latin Hypercube Sampling"""

    def sample_continuous(self, numsamples: int, doe: DesignSpace) -> np.ndarray:
        continuous = doe.get_continuous_parameters()
        problem = {
            "num_vars": len(continuous),
            "names": [s.name for s in continuous],
            "bounds": [[s.lower_bound, s.upper_bound] for s in continuous],
        }

        samples = latin.sample(problem, N=numsamples, seed=self.seed)
        return samples
