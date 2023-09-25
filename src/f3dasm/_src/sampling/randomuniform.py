"""
Random uniform Sampling
Reference: `numpy.random.uniform <https://numpy.org/doc/stable/reference/random/generated/numpy.random.uniform.html>`_
"""

#                                                                       Modules
# =============================================================================

# Third-party core
import numpy as np

# Locals
from .sampler import Sampler

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class RandomUniform(Sampler):
    """
    Sampling via random uniform sampling
    """

    def sample_continuous(self, numsamples: int) -> np.ndarray:
        """Sample from continuous space

        Parameters
        ----------
        numsamples
            number of samples

        Returns
        -------
            samples
        """
        continuous = self.domain.continuous
        samples = np.random.uniform(size=(numsamples, len(continuous)))

        # stretch samples
        samples = self._stretch_samples(samples)
        return samples