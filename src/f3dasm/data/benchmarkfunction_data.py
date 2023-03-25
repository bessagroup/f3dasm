#                                                                       Modules
# =============================================================================

# Standard
from typing import Union

# Third-party
import numpy as np

# Locals
from ..design import DesignSpace, ExperimentData, make_nd_continuous_design
from ..functions import Function, PyBenchFunction
from ..sampling import RandomUniform
from .learningdata import LearningData

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class BenchmarkFunctionData(LearningData):
    def __init__(self, n: int, dimensionality: int, bounds: np.ndarray,
                 function_class: PyBenchFunction, noise: Union[None, int] = None, seed: Union[None, int] = None):
        """Create noisy sampled points of a benchmarkfunction

        Parameters
        ----------
        n
            number of points
        dimensionality
            number of dimensions
        bounds
            box-constraint boundaries of the sampling points
        function_class
            PyBench function
        noise, optional
            standard deviation of the noise, by default None
        seed, optional
            seed for the random number generator, by default None
        """

        self.n = n
        self.dimensionality = dimensionality
        self.bounds = bounds
        self.noise = noise
        self.seed = seed

        self.function = self._create_function_instance(function_class)
        self._create_datapoints()

    def _create_function_instance(self, function_class) -> PyBenchFunction:
        return function_class(dimensionality=self.dimensionality,
                              scale_bounds=self.bounds, noise=self.noise, seed=self.seed, offset=False)

    def _create_datapoints(self):
        design: DesignSpace = make_nd_continuous_design(bounds=self.bounds, dimensionality=self.dimensionality)
        sampler = RandomUniform(design=design)
        data: ExperimentData = sampler.get_samples(numsamples=self.n)

        data.add_output(output=self.function(data))

        self.X = data.get_input_data()
        self.y = data.get_output_data()

    def get_input_data(self) -> np.ndarray:
        return self.X.to_numpy()

    def get_labels(self) -> np.ndarray:
        return self.y.to_numpy()
