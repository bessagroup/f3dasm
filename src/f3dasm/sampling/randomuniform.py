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
    """Sampling via random uniform sampling"""

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
        continuous = self.design.get_continuous_input_parameters()
        dimensions = len(continuous)

        samples = np.random.uniform(size=(numsamples, dimensions))

        # stretch samples
        samples = self._stretch_samples(samples)
        return samples
