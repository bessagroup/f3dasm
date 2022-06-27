import numpy as np
from SALib.sample import latin

from f3dasm.src.designofexperiments import DoE

from f3dasm.src.samplingmethod import SamplingMethod


class LatinHypercube(SamplingMethod):
    """Sampling via Latin Hypercube Sampling"""

    def sample_continuous(self, numsamples: int, doe: DoE) -> np.ndarray:
        continuous = doe.get_continuous_parameters()
        problem = {
            "num_vars": len(continuous),
            "names": [s.name for s in continuous],
            "bounds": [[s.lower_bound, s.upper_bound] for s in continuous],
        }

        samples = latin.sample(problem, N=numsamples, seed=self.seed)
        return samples
