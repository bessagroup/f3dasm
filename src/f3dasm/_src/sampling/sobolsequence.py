"""Sobol Sequence Sampling"""

#                                                                       Modules
# =============================================================================

# Third-party
import numpy as np
from SALib.sample import sobol_sequence

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


class SobolSequence(Sampler):
    """Sampling via Sobol Sequencing with SALib

    Reference: `SALib <https://salib.readthedocs.io/en/latest/>`_"""

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

        samples = sobol_sequence.sample(numsamples, len(continuous))

        # stretch samples
        samples = self._stretch_samples(samples)
        return samples
