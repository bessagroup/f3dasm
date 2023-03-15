#                                                                       Modules
# =============================================================================

# Standard
from typing import Any, Union

# Third-party core
import numpy as np

# Locals
from .._imports import try_import
from ..design import DesignSpace
from .sampler import Sampler

# Third-party extension
with try_import('sampling') as _imports:
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

    def __init__(self, design: DesignSpace, seed: Union[Any, int] = None):
        _imports.check()
        super().__init__(design, seed)

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
