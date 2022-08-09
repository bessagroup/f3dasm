from typing import List
import numpy as np
from .base.data import Data

from .base.optimization import Optimizer
from .base.samplingmethod import SamplingInterface
from .base.function import Function


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

    samples.add_output(output=function.__call__(samples), label="y")

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

    # TODO: Make sure Data in optimizer is initialized everytime new realization is taken
    # Create a random seed
    seed = np.random.randint(low=0, high=1e5)
    all_data = []

    for _ in range(realizations):
        data = run_optimization(
            optimizer=optimizer, function=function, sampler=sampler, iterations=iterations, seed=seed
        )
        all_data.append(data)

        # Increase seed
        seed += 1

    return all_data
