#                                                                       Modules
# =============================================================================

# Standard
from typing import Optional

# Third-party
import autograd.numpy as np

# Locals
from ....experimentdata import ExperimentData
from ..function import Function
from .augmentor import (EmptyAugmentor, FunctionAugmentor, Noise, Offset,
                        Scale, _Augmentor)

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
        self.scale_bounds = scale_bounds
        self.noise = noise
        self.offset = offset

    def arm(self, data: ExperimentData):
        self.set_seed(self.seed)
        self.augmentor = FunctionAugmentor()
        self.dimensionality = len(data.domain)
        self._set_parameters()
        s = self._configure_scale_bounds()
        n = self._configure_noise()
        o = self._configure_offset()

        self.augmentor = FunctionAugmentor(input_augmentors=[o, s],
                                           output_augmentors=[n])

    def _configure_scale_bounds(self) -> _Augmentor:
        """Create a Scale augmentor"""
        if self.scale_bounds is None:
            return EmptyAugmentor()

        s = Scale(scale_bounds=self.scale_bounds,
                  input_domain=self.input_domain)
        self.augmentor = FunctionAugmentor(input_augmentors=[s])
        return s

        # self.augmentor.add_input_augmentor(s)

    def _configure_noise(self):
        """Create a Noise augmentor"""
        if self.noise is None:
            return EmptyAugmentor()

        n = Noise(noise=self.noise, rng=self.rng)
        return n

    def _configure_offset(self):
        """Create an Offset augmentor"""
        if not self.offset or self.scale_bounds is None:
            return EmptyAugmentor()

        g = self._get_global_minimum_for_offset_calculation()
        unscaled_offset = np.atleast_1d(
            [
                # This is added so we only create offsets in one quadrant
                self.rng.uniform(
                    low=-abs(g[d] - self.scale_bounds[d, 0]),
                    high=0.0)
                for d in range(self.dimensionality)
            ]
        )

        return Offset(offset=unscaled_offset)
        # self.augmentor.insert_input_augmentor(position=0, augmentor=self.o)

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
