import time
from typing import List
import numpy as np
from pkg_resources import resource_listdir
from .base.data import Data

from .base.optimization import Optimizer
from .base.samplingmethod import SamplingInterface
from .base.function import Function

from multiprocessing import Pool


def run_optimization(
    optimizer: Optimizer, function: Function, sampler: SamplingInterface, iterations: int, seed: int
) -> Data:
    """Run optimization on some benchmark function"""

    # Set function seed
    function.set_seed(seed)
    optimizer.set_seed(seed)
    sampler.set_seed(seed)

    # Sample
    samples = sampler.get_samples(numsamples=optimizer.hyperparameters["population"])

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
) -> List[Data]:
    """Run multiple realizations of the same algorithm on a benchmark function"""
    # start_t = time.perf_counter()

    seed = np.random.randint(low=0, high=1e5)
    args = [(optimizer, function, sampler, iterations, seed + index) for index, _ in enumerate(range(realizations))]

    with Pool() as pool:
        results = pool.starmap(run_optimization, args)  # maybe implement pool.starmap_async ?

    # end_t = time.perf_counter()

    # total_duration = end_t - start_t
    # print(f"Optimization took {total_duration:.2f}s total")

    return results
