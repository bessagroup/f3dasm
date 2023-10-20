"""
Module to optimize benchmark optimization functions
"""
#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional

# Third-party
import numpy as np
import pandas as pd
import xarray as xr
from pathos.helpers import mp

from f3dasm.design import Domain
from f3dasm.optimization import Optimizer

# Locals
from .datageneration.datagenerator import DataGenerator
from .datageneration.functions.function_factory import datagenerator_factory
from .experimentdata.experimentdata import ExperimentData
from .logger import logger, time_and_log
from .optimization.optimizer_factory import optimizer_factory

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class OptimizationResult:
    def __init__(self, data: List[ExperimentData], optimizer: Optimizer,
                 kwargs: Optional[Dict[str, Any]], data_generator: DataGenerator,
                 number_of_samples: int, seeds: List[int], opt_time: float = 0.0):
        """Optimization results object

        Parameters
        ----------
        data
            Data objects for each realization
        optimizer
            classname of the optimizer used
        data_generator
            the data_generator to get objective values
        kwargs
            the kwargs used for the data_generator
        number_of_samples
            number of initial samples, sampled by the sampling strategy
        seeds
            list of seeds that were used for each realization
        opt_time
            total optimization time
        """
        self.data = data
        self.optimizer = optimizer_factory(optimizer=optimizer, domain=self.data[0].domain)
        self.data_generator = data_generator
        self.kwargs = kwargs,
        self.number_of_samples = number_of_samples
        self.seeds = seeds
        self.opt_time = opt_time

        self.func = datagenerator_factory(data_generator=self.data_generator,
                                          domain=self.data[0].domain, kwargs=kwargs)
        self._log()

    def _log(self):
        # Log
        logger.info(
            (f"Optimized {self.data_generator} function (seed={self.func.seed}, "
             f"dim={len(self.data[0].domain)}, "
             f"noise={self.func.noise}) "
             f"with {self.optimizer.get_name()} optimizer for "
             f"{len(self.data)} realizations ({self.opt_time:.3f} s).")
        )

    def to_xarray(self) -> xr.Dataset:
        xarr = xr.concat([realization.to_xarray() for realization in self.data],
                         dim=xr.DataArray(np.arange(len(self.data)), dims='realization'))

        xarr.attrs['number_of_samples']: int = self.number_of_samples
        xarr.attrs['realization_seeds']: List[int] = list(self.seeds)

        # Benchmark functions
        xarr.attrs['function_seed']: int = self.func.seed
        xarr.attrs['function_name']: str = self.data_generator
        xarr.attrs['function_noise']: str = self.func.noise
        xarr.attrs['function_dimensionality']: int = len(self.data[0].domain)

        # Global minimum function
        _, g = self.func.get_global_minimum(d=self.func.dimensionality)
        xarr.attrs['function_global_minimum']: float = float(np.array(g if not isinstance(g, list) else g[0])[0, 0])
        return xarr


def run_optimization(
    optimizer: Optimizer | str,
    data_generator: DataGenerator | str,
    sampler: Callable | str,
    domain: Domain,
    iterations: int,
    seed: int,
    kwargs: Optional[Dict[str, Any]] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
    number_of_samples: int = 30,
) -> ExperimentData:
    """Run optimization on some benchmark function

    Parameters
    ----------
    optimizer
        the optimizer used
    data_generator
        the data_generator to get objective values
    sampler
        the sampling strategy
    domain
        the domain
    iterations
        number of iterations
    seed
        seed for the random number generator
    kwargs
        additional keyword arguments for the data generator
    hyperparameters
        additional keyword arguments for the optimizer
    number_of_samples, optional
        number of initial samples, sampled by the sampling strategy

    Returns
    -------
        Data object with the optimization data results
    """
    if kwargs is None:
        kwargs = {}

    if hyperparameters is None:
        hyperparameters = {}

    # Set function seed
    optimizer = optimizer_factory(optimizer=optimizer, domain=domain, hyperparameters=hyperparameters)

    optimizer.set_seed()

    # Sample
    data = ExperimentData.from_sampling(sampler=sampler, domain=domain, n_samples=number_of_samples, seed=seed)

    data.evaluate(data_generator, mode='sequential', kwargs=kwargs)
    data.optimize(optimizer=optimizer, data_generator=data_generator,
                  iterations=iterations, kwargs=kwargs, hyperparameters=hyperparameters)

    return data


# def run_optimization_to_disk(
#     optimizer: Optimizer,
#     data_generator: DataGenerator,
#     sampler: Sampler,
#     iterations: int,
#     seed: int,
#     number_of_samples: int = 30,
#     realization_index: int = 0,
# ) -> None:

#     # Set function seed
#     optimizer.set_seed(seed)
#     sampler.set_seed(seed)

#     # Sample
#     data = sampler.get_samples(numsamples=number_of_samples)

#     data.evaluate(data_generator, mode='sequential')
#     data.optimize(optimizer=optimizer, data_generator=data_generator, iterations=iterations)

#     # TODO: .get_name() method is not implemented for DataGenerator base class
#     data.to_xarray().to_netcdf(
#         f'{data_generator.get_name()}_{optimizer.get_name()}_{seed-realization_index}_{realization_index}.temp')


@time_and_log
def run_multiple_realizations(
    optimizer: Optimizer,
    data_generator: DataGenerator | str,
    sampler: Callable | str,
    domain: Domain,
    iterations: int,
    realizations: int,
    kwargs: Optional[Dict[str, Any]] = None,
    hyperparameters: Optional[Dict[str, Any]] = None,
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
    data_generator
        the data_generator to get objective values
    sampler
        the sampling strategy
    domain
        the domain
    iterations
        number of iterations
    realizations
        number of realizations
    kwargs
        additional keyword arguments for the data generator
    hyperparameters
        additional keyword arguments for the optimizer
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

    start_timer = perf_counter()

    if kwargs is None:
        kwargs = {}

    if hyperparameters is None:
        hyperparameters = {}

    if seed is None:
        seed = np.random.randint(low=0, high=1e5)

    if parallelization:
        args = [
            (optimizer, data_generator, sampler, domain, iterations,
             seed + index, kwargs, hyperparameters, number_of_samples)
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
                "data_generator": data_generator,
                "sampler": sampler,
                "domain": domain,
                "iterations": iterations,
                "kwargs": kwargs,
                "hyperparameters": hyperparameters,
                "number_of_samples": number_of_samples,
                "seed": seed + index,
            }
            results.append(run_optimization(**args))

    opt_time = perf_counter() - start_timer

    return OptimizationResult(
        data=results,
        optimizer=optimizer,
        data_generator=data_generator,
        kwargs=kwargs,
        number_of_samples=number_of_samples,
        seeds=[seed + i for i in range(realizations)],
        opt_time=opt_time,
    )


# @time_and_log
# def run_multiple_realizations_to_disk(
#     optimizer: Optimizer,
#     data_generator: DataGenerator,
#     sampler: Sampler,
#     iterations: int,
#     realizations: int,
#     number_of_samples: int = 30,
#     parallelization: bool = True,
#     verbal: bool = False,
#     seed: int or Any = None,
# ) -> OptimizationResult:

#     if seed is None:
#         seed = np.random.randint(low=0, high=1e5)

#     if parallelization:
#         args = [
#             (optimizer, data_generator, sampler, iterations,
#              seed + index, number_of_samples, index)
#             for index, _ in enumerate(range(realizations))
#         ]

#         with mp.Pool() as pool:
#             # maybe implement pool.starmap_async ?
#             results = pool.starmap(run_optimization_to_disk, args)

#     else:
#         results = []
#         for index in range(realizations):
#             args = {
#                 "optimizer": optimizer,
#                 "data_generator": data_generator,
#                 "sampler": sampler,
#                 "iterations": iterations,
#                 "number_of_samples": number_of_samples,
#                 "seed": seed + index,
#                 "realization_index": index,
#             }
#             results.append(run_optimization_to_disk(**args))

#     files = f'{data_generator.get_name()}_{optimizer.get_name()}_{seed}_*.temp'
#     combined_dataset = xr.open_mfdataset(files, combine="nested", parallel=True, concat_dim=xr.DataArray(
#         range(realizations), dims='realization'))

#     combined_dataset.to_netcdf(f'{data_generator.get_name()}_{optimizer.get_name()}_combined_{seed}.temp')

#     # remove all the files
#     for file in Path().glob(files):
#         file.unlink(missing_ok=True)


def calculate_mean_std(results: OptimizationResult):  # OptimizationResult
    mean_y = pd.concat([d.output_data.to_dataframe().cummin()
                       for d in results.data], axis=1).mean(axis=1)
    std_y = pd.concat([d.output_data.to_dataframe().cummin()
                      for d in results.data], axis=1).std(axis=1)
    return mean_y, std_y
