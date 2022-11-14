import autograd.numpy as np

from ..base.samplingmethod import SamplingInterface


class RandomUniform(SamplingInterface):
    """Sampling via random uniform sampling"""

    def sample_continuous(self, numsamples: int) -> np.ndarray:
        """Sample from continuous space

        Parameters
        ----------
        numsamples
            numeber of samples

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
