#                                                                       Modules
# =============================================================================

# Third-party
import autograd.numpy as np

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
    from torch.quasirandom import SobolEngine


#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Leo Guo (L.L.Guo@tudelft.nl)'
__credits__ = ['Leo Guo', 'Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class SobolSequence_torch(Sampler):
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

        sobolengine = SobolEngine(dimension=dimensions, scramble=True, seed=self.seed)
        samples = sobolengine.draw(numsamples).numpy()

        # stretch samples
        samples = self._stretch_samples(samples)
        return samples