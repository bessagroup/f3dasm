from typing import Any
import numpy as np
from SALib.sample import latin

from f3dasm.src.designofexperiments import DoE

from ..src.samplingmethod import SamplingMethod


class LatinHypercube(SamplingMethod):
    """Sampling via Latin Hypercube Sampling"""

    def sample_continuous(self, numsamples: int, doe: DoE) -> np.array:
        continuous = doe.getContinuousParameters()
        problem = {
            "num_vars": len(continuous),
            "names": [s.name for s in continuous],
            "bounds": [[s.lower_bound, s.upper_bound] for s in continuous],
        }

        samples = latin.sample(problem, N=numsamples, seed=self.seed)
        return samples
