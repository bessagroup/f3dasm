"""
Module to optimize benchmark optimization functions
"""
#                                                                       Modules
# =============================================================================

# Standard
import json
import logging
import time
from typing import Any, List

# Third-party
import numpy as np
import pandas as pd
from pathos.helpers import mp

from f3dasm.optimization import Optimizer, create_optimizer_from_json
from f3dasm.sampling import Sampler, create_sampler_from_json

# Locals
from .design import ExperimentData, create_experimentdata_from_json
from .functions import create_function_from_json
from .functions.function import Function

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

    def to_json(self):
        args = {'data': [d.to_json() for d in self.data],
                'optimizer': self.optimizer.to_json(),
                'function': self.function.to_json(),
                'sampler': self.sampler.to_json(),
                'number_of_samples': self.number_of_samples,
                'seeds': self.seeds
                }

        return json.dumps(args)

    def _log(self):
        # Log
        logging.info(
            (f"Optimized {self.function.get_name()} function (seed={self.function.seed}, "
             f"dim={self.function.dimensionality}, "
             f"with {self.optimizer.get_name()} optimizer for "
             f"{len(self.data)} realizations!")
        )


def create_optimizationresult_from_json(json_string: str) -> OptimizationResult:
    optimizationresult_dict = json.loads(json_string)
    return _create_optimizationresult_from_dict(optimizationresult_dict)


def _create_optimizationresult_from_dict(optimizationresult_dict: dict) -> OptimizationResult:
    args = {
        'data': [create_experimentdata_from_json(json_data) for json_data in optimizationresult_dict['data']],
        'optimizer': create_optimizer_from_json(optimizationresult_dict['optimizer']),
        'function': create_function_from_json(optimizationresult_dict['function']),
        'sampler': create_sampler_from_json(optimizationresult_dict['sampler']),
        'number_of_samples': optimizationresult_dict['number_of_samples'],
        'seeds': optimizationresult_dict['seeds'],
    }
    return OptimizationResult(**args)


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

    samples.add_output(output=function(samples), label="y")

    optimizer.set_data(samples)

    # Iterate
    optimizer.iterate(iterations=iterations, function=function)
    res = optimizer.extract_data()

    # Reset the parameters
    optimizer.__post_init__()

    # Reset data
    optimizer.data.reset_data()

    return res


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
    start_t = time.perf_counter()

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

    end_t = time.perf_counter()

    total_duration = end_t - start_t
    if verbal:
        print(f"Optimization took {total_duration:.2f}s total")

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
