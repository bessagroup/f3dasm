"""
Module to optimize benchmark optimization functions
"""
#                                                                       Modules
# =============================================================================

# Standard
import json
import time
from typing import Any, List, Type

# Third-party
import numpy as np
import pandas as pd
import xarray as xr
from pathos.helpers import mp

from f3dasm.optimization import Optimizer
from f3dasm.sampling import Sampler

from .datageneration.functions import create_function_from_json
from .datageneration.functions.function import Function
# Locals
from .design import ExperimentData
from .logger import logger, time_and_log

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class OptimizationResult:
    def __init__(self, data: List[ExperimentData], optimizer: Optimizer, function: Function,
                 sampler: Sampler, number_of_samples: int, seeds: List[int]):
        """Optimization results object

        Parameters
        ----------
        data
            Data objects for each realization
        optimizer
            classname of the optimizer used
        function
            functionname that was optimized
        sampler
            classname of the initial sampling strategy
        number_of_samples
            number of initial samples, sampled by the sampling strategy
        seeds
            list of seeds that were used for each realization
        """
        self.data = data
        self.optimizer = optimizer
        self.function = function
        self.sampler = sampler
        self.number_of_samples = number_of_samples
        self.seeds = seeds
        self._log()

    def _log(self):
        # Log
        logger.info(
            (f"Optimized {self.function.get_name()} function (seed={self.function.seed}, "
             f"dim={self.function.dimensionality}, "
             f"noise={self.function.noise}) "
             f"with {self.optimizer.get_name()} optimizer for "
             f"{len(self.data)} realizations.")
        )

    def to_xarray(self) -> xr.Dataset:
        xarr = xr.concat([realization.to_xarray() for realization in self.data],
                         dim=xr.DataArray(np.arange(len(self.data)), dims='realization'))

        xarr.attrs['number_of_samples']: int = self.number_of_samples
        xarr.attrs['realization_seeds']: List[int] = list(self.seeds)

        # Benchmark functions
        xarr.attrs['function_seed']: int = self.function.seed
        xarr.attrs['function_name']: str = self.function.get_name()
        xarr.attrs['function_noise']: str = self.function.noise
        xarr.attrs['function_dimensionality']: int = self.function.dimensionality

        # Global minimum function
        _, g = self.function.get_global_minimum(d=self.function.dimensionality)
        xarr.attrs['function_global_minimum']: float = float(np.array(g if not isinstance(g, list) else g[0])[0, 0])
        return xarr


def run_optimization(
    optimizer: Optimizer,
    function: Function,
    sampler: Sampler,
    iterations: int,
    seed: int,
    number_of_samples: int = 30,
) -> ExperimentData:
    """Run optimization on some benchmark function

    Parameters
    ----------
    optimizer
        the optimizer used
    function
        the function to be optimized
    sampler
        the sampling strategy
    iterations
        number of iterations
    seed
        seed for the random number generator
    number_of_samples, optional
        number of initial samples, sampled by the sampling strategy

    Returns
    -------
        Data object with the optimization data results
    """

    # Set function seed
    # function.set_seed(seed)
    optimizer.set_seed(seed)
    sampler.set_seed(seed)

    # Sample
    samples = sampler.get_samples(numsamples=number_of_samples)

    samples.fill_output(output=function(samples.get_input_data().to_numpy()), label="y")

    optimizer.set_data(samples)

    # Iterate
    optimizer.iterate(iterations=iterations, function=function)
    res = optimizer.extract_data()

    # Reset the parameters
    optimizer.__post_init__()

    # Reset data
    optimizer.data.reset_data()

    return res


@time_and_log
def run_multiple_realizations(
    optimizer: Optimizer,
    function: Function,
    sampler: Sampler,
    iterations: int,
    realizations: int,
    number_of_samples: int = 30,
    parallelization: bool = True,
    verbal: bool = False,
    seed: int or Any = None,
) -> OptimizationResult:
    """Run multiple realizations of the same algorithm on a benchmark function

    Parameters
    ----------
    optimizer
        the optimizer used
    function
        the function to be optimized
    sampler
        the sampling strategy
    iterations
        number of iterations
    realizations
        number of realizations
    number_of_samples, optional
        number of initial samples, sampled by the sampling strategy
    parallelization, optional
        set True to enable parallel execution of each realization
    verbal, optional
        set True to have more debug message
    seed, optional
        seed for the random number generator

    Returns
    -------
        Object with the optimization data results
    """
    if seed is None:
        seed = np.random.randint(low=0, high=1e5)

    if parallelization:
        args = [
            (optimizer, function, sampler, iterations,
             seed + index, number_of_samples)
            for index, _ in enumerate(range(realizations))
        ]

        with mp.Pool() as pool:
            # maybe implement pool.starmap_async ?
            results = pool.starmap(run_optimization, args)

    else:
        results = []
        for index in range(realizations):
            args = {
                "optimizer": optimizer,
                "function": function,
                "sampler": sampler,
                "iterations": iterations,
                "number_of_samples": number_of_samples,
                "seed": seed + index,
            }
            results.append(run_optimization(**args))

    return OptimizationResult(
        data=results,
        optimizer=optimizer,
        function=function,
        sampler=sampler,
        number_of_samples=number_of_samples,
        seeds=[seed + i for i in range(realizations)],
    )


def calculate_mean_std(results):  # OptimizationResult
    mean_y = pd.concat([d.get_output_data().cummin()
                       for d in results.data], axis=1).mean(axis=1)
    std_y = pd.concat([d.get_output_data().cummin()
                      for d in results.data], axis=1).std(axis=1)
    return mean_y, std_y
