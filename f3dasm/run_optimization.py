from dataclasses import dataclass
import logging
import time
from typing import Any, List
import numpy as np

# import f3dasm

from f3dasm.base.data import Data

from f3dasm.base.optimization import Optimizer
from f3dasm.base.samplingmethod import SamplingInterface
from f3dasm.base.function import Function

from pathos.helpers import mp


@dataclass
class OptimizationResult:
    data: List[Data]
    optimizer: str
    hyperparameters: dict
    function: Function
    sampler: str
    number_of_samples: int
    seeds: List[int]

    def __post_init__(self):
        # Log
        logging.info(
            f"Optimized {self.function.get_name()} function (seed={self.function.seed}, dim={self.function.dimensionality}) with {self.optimizer} optimizer for {len(self.data)} realizations!"
        )


def run_optimization(
    optimizer: Optimizer,
    function: Function,
    sampler: SamplingInterface,
    iterations: int,
    seed: int,
    number_of_samples: int = 30,
) -> Data:
    """Run optimization on some benchmark function"""

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
    sampler: SamplingInterface,
    iterations: int,
    realizations: int,
    number_of_samples: int = 30,
    parallelization: bool = True,
    verbal: bool = False,
    seed: int or Any = None,
) -> OptimizationResult:
    """Run multiple realizations of the same algorithm on a benchmark function"""
    start_t = time.perf_counter()

    if seed is None:
        seed = np.random.randint(low=0, high=1e5)

    if parallelization:
        args = [
            (optimizer, function, sampler, iterations, seed + index, number_of_samples)
            for index, _ in enumerate(range(realizations))
        ]

        with mp.Pool() as pool:
            results = pool.starmap(run_optimization, args)  # maybe implement pool.starmap_async ?

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
        optimizer=optimizer.get_name(),
        hyperparameters=optimizer.parameter,
        function=function,
        sampler=sampler,
        number_of_samples=number_of_samples,
        seeds=[seed + i for i in range(realizations)],
    )
