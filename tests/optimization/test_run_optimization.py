from __future__ import annotations

import os
# Standard
from time import perf_counter
from typing import Any, Callable, Dict, List, Optional

# Third-party
import numpy as np
import pytest
import xarray as xr
from pathos.helpers import mp

# Locals
from f3dasm import ExperimentData, logger
from f3dasm._src.datageneration.functions.function_factory import \
    _datagenerator_factory
from f3dasm._src.optimization.optimizer_factory import _optimizer_factory
from f3dasm.datageneration import DataGenerator
from f3dasm.datageneration.functions import FUNCTIONS_2D, FUNCTIONS_7D
from f3dasm.design import Domain, make_nd_continuous_domain
from f3dasm.optimization import OPTIMIZERS, Optimizer


class OptimizationResult:
    def __init__(self, data: List[ExperimentData], optimizer: Optimizer,
                 kwargs: Optional[Dict[str, Any]],
                 data_generator: DataGenerator,
                 number_of_samples: int, seeds: List[int],
                 opt_time: float = 0.0):
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
        self.data_generator = data_generator
        self.kwargs = kwargs,
        self.number_of_samples = number_of_samples
        self.seeds = seeds
        self.opt_time = opt_time

        self.func = _datagenerator_factory(
            data_generator=self.data_generator,
            domain=self.data[0].domain, kwargs=kwargs)
        self.optimizer = _optimizer_factory(
            optimizer=optimizer, domain=self.data[0].domain,
            data_generator=self.func)
        self._log()

    def _log(self):
        # Log
        logger.info(
            (f"Optimized {self.data_generator} \
                function (seed={self.func.seed}, "
             f"dim={len(self.data[0].domain)}, "
             f"noise={self.func.noise}) "
             f"with {self.optimizer.__class__.__name__} optimizer for "
             f"{len(self.data)} realizations ({self.opt_time:.3f} s).")
        )

    def to_xarray(self) -> xr.Dataset:
        xarr = xr.concat(
            [realization.to_xarray()
             for realization in self.data],
            dim=xr.DataArray(
                np.arange(len(self.data)), dims='realization'))

        xarr.attrs['number_of_samples']: int = self.number_of_samples
        xarr.attrs['realization_seeds']: List[int] = list(self.seeds)

        # Benchmark functions
        xarr.attrs['function_seed']: int = self.func.seed
        xarr.attrs['function_name']: str = self.data_generator
        xarr.attrs['function_noise']: str = self.func.noise
        xarr.attrs['function_dimensionality']: int = len(self.data[0].domain)

        # Global minimum function
        _, g = self.func.get_global_minimum(d=self.func.dimensionality)
        xarr.attrs['function_global_minimum']: float = float(
            np.array(g if not isinstance(g, list) else g[0])[0, 0])
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
    data_generator = _datagenerator_factory(
        data_generator=data_generator, domain=domain, kwargs=kwargs)

    optimizer = _optimizer_factory(
        optimizer=optimizer, domain=domain, data_generator=data_generator,
        hyperparameters=hyperparameters)

    # Sample
    data = ExperimentData.from_sampling(
        sampler=sampler, domain=domain, n_samples=number_of_samples, seed=seed)

    data.evaluate(data_generator, mode='sequential', kwargs=kwargs)
    data.optimize(optimizer=optimizer, data_generator=data_generator,
                  iterations=iterations, kwargs=kwargs,
                  hyperparameters=hyperparameters)

    return data


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
    seed: int | Any = None,
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


@pytest.mark.smoke
@pytest.mark.parametrize("optimizer", OPTIMIZERS)
@pytest.mark.parametrize("data_generator", ['Levy', 'Ackley', 'Sphere'])
@pytest.mark.parametrize("dimensionality", [2])
def test_run_multiple_realizations_3_functions(data_generator: str,
                                               optimizer: str, dimensionality: int):
    test_run_multiple_realizations(data_generator, optimizer, dimensionality)


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
@pytest.mark.parametrize("data_generator", FUNCTIONS_2D)
@pytest.mark.parametrize("dimensionality", [2])
def test_run_multiple_realizations(data_generator: str, optimizer: str, dimensionality: int):
    iterations = 30
    realizations = 3
    domain = np.tile([0.0, 1.0], (dimensionality, 1))

    domain = make_nd_continuous_domain(
        dimensionality=dimensionality, bounds=domain)

    kwargs = {'scale_bounds': domain.get_bounds()}
    sampler = 'random'

    # Check if os is windows
    if os.name == 'nt':
        PARALLELIZATION = False
    else:
        PARALLELIZATION = True

    if optimizer in ['EvoSaxCMAES', 'EvoSaxSimAnneal', 'EvoSaxPSO', 'EvoSaxDE']:
        PARALLELIZATION = False

    _ = run_multiple_realizations(
        optimizer=optimizer,
        data_generator=data_generator,
        kwargs=kwargs,
        sampler=sampler,
        domain=domain,
        iterations=iterations,
        realizations=realizations,
        parallelization=PARALLELIZATION,
    )


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
@pytest.mark.parametrize("data_generator", FUNCTIONS_7D)
@pytest.mark.parametrize("dimensionality", [7])
def test_run_multiple_realizations_7D(data_generator: str, optimizer: str, dimensionality: int):
    test_run_multiple_realizations(data_generator, optimizer, dimensionality)


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
@pytest.mark.parametrize("data_generator", ['griewank'])
@pytest.mark.parametrize("dimensionality", [2])
def test_run_multiple_realizations_fast(data_generator: str, optimizer: str, dimensionality: int):
    test_run_multiple_realizations(data_generator, optimizer, dimensionality)


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
