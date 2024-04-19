#                                                                       Modules
# =============================================================================

# Standard
from typing import Optional

# Third-party
import autograd.numpy as np

# Locals
from ..function import Function
from .augmentor import Noise, Offset, Scale

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling', 'Axel Thevenot']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class PyBenchFunction(Function):
    def __init__(
            self,
            dimensionality: int,
            scale_bounds: Optional[np.ndarray] = None,
            noise: Optional[float] = None,
            offset: bool = True,
            seed: Optional[int] = None):
        """Adapter for pybenchfunction, created by Axel Thevenot (2020).
        Github repository:
         https://github.com/AxelThevenot/Python_Benchmark_Test_
        Optimization_Function_Single_Objective

        Parameters
        ----------
        dimensionality
            number of dimensions
        scale_bounds, optional
            array containing the lower and upper bound of the scaling
             factor of the input data, by default None
        noise, optional
            inflict Gaussian noise on the input, by default None
        offset, optional
            set this True to randomly off-set the pybenchfunction,
             by default True
        seed, optional
            seed for the random number generator, by default None
        """
        super().__init__(seed=seed)
        self.dimensionality = dimensionality
        self.scale_bounds = scale_bounds
        self.noise = noise
        self.offset = offset

        self.__post_init__()

    def __post_init__(self):
        self._set_parameters()

        self._configure_scale_bounds()
        self._configure_noise()
        self._configure_offset()

    def _configure_scale_bounds(self):
        """Create a Scale augmentor"""
        if self.scale_bounds is None:
            return
        s = Scale(scale_bounds=self.scale_bounds,
                  input_domain=self.input_domain)
        self.augmentor.add_input_augmentor(s)

    def _configure_noise(self):
        """Create a Noise augmentor"""
        if self.noise is None:
            return

        n = Noise(noise=self.noise)
        self.augmentor.add_output_augmentor(n)

    def _configure_offset(self):
        """Create an Offset augmentor"""
        if not self.offset or self.scale_bounds is None:
            return

        g = self._get_global_minimum_for_offset_calculation()

        unscaled_offset = np.atleast_1d(
            [
                # np.random.uniform(
                #     low=-abs(g[d] - self.scale_bounds[d, 0]),
                #     high=abs(g[d] - self.scale_bounds[d, 1]))

                # This is added so we only create offsets in one quadrant

                np.random.uniform(
                    low=-abs(g[d] - self.scale_bounds[d, 0]),
                    high=0.0)
                for d in range(self.dimensionality)
            ]
        )

        self.o = Offset(offset=unscaled_offset)
        self.augmentor.insert_input_augmentor(position=0, augmentor=self.o)

    def _get_global_minimum_for_offset_calculation(self):
        """Get the global minimum used for offset calculations

        Returns
        -------
            Returns a numpy array containing the global
            minimum or all zeroes if no global minimum exists
        """
        global_minimum_method = getattr(self, "get_global_minimum", None)
        if callable(global_minimum_method):
            g = self.get_global_minimum(d=self.dimensionality)[0]

            if g is None:
                g = np.zeros(self.dimensionality)

            if g.ndim == 2:
                g = g[0]

        else:
            g = np.zeros(self.dimensionality)

        return g

    def _set_parameters(self):
        """Function where certain parameters of pybenchfunctions are set.
        Inhereted by subclasses"""
        ...
