#                                                                       Modules
# =============================================================================

# Locals
from .._imports import try_import
from .sampler import Sampler

# Third-party
with try_import('sampling') as _imports:
    import autograd.numpy as np
    from SALib.sample import sobol_sequence


#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class SobolSequence(Sampler):
    """Sampling via Sobol Sequencing with SALib"""

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

        samples = sobol_sequence.sample(numsamples, dimensions)

        # stretch samples
        samples = self._stretch_samples(samples)
        return samples
