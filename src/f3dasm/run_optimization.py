#                                                                       Modules
# =============================================================================

# Standard
import logging
import time
from dataclasses import dataclass
from typing import Any, List

# Third-party
import numpy as np
import pandas as pd
from pathos.helpers import mp
from sklearn import preprocessing

# Locals
from .base.data import Data
from .base.function import Function
from .base.optimization import Optimizer
from .base.samplingmethod import SamplingInterface
from .base.utils import calculate_mean_std

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


@dataclass
class OptimizationResult:
    """Optimization results object

    :param data: Data objects for each realization
    :param optimizer: classname of the optimizer used
    :param hyperparameters: hyperparameters of the optimizer
    :param function: function that was optimized
    :param sampler: classname of the initial sampling strategy
    :param number_of_samples: number of initial samples, sampled by the sampling strategy
    :param seeds: list of seeds that were used for each realization
    """

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
            f"Optimized {self.function.get_name()} function (seed={self.function.seed}, \
            dim={self.function.dimensionality}, noise={self.function.noise}) with {self.optimizer} \
            optimizer for {len(self.data)} realizations!"
        )


def run_optimization(
    optimizer: Optimizer,
    function: Function,
    sampler: SamplingInterface,
    iterations: int,
    seed: int,
    number_of_samples: int = 30,
) -> Data:
    """Run optimization on some benchmark function

    :param optimizer: the optimizer used
    :param function: the function to be optimized
    :param sampler: the sampling strategy
    :param iterations: number of iterations
    :param seed: seed for the random number generator
    :param number_of_samples: number of initial samples, sampled by the sampling strategy
    :return: Data object with the optimization data results
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
    sampler: SamplingInterface,
    iterations: int,
    realizations: int,
    number_of_samples: int = 30,
    parallelization: bool = True,
    verbal: bool = False,
    seed: int or Any = None,
) -> OptimizationResult:
    """Run multiple realizations of the same algorithm on a benchmark function

    :param optimizer: the optimizer used
    :param function: the function to be optimized
    :param sampler: the sampling strategy
    :param iterations: number of iterations
    :param realizations: number of realizations
    :param number_of_samples: number of initial samples, sampled by the sampling strategy
    :param parallelization: set True to enable parallelization
    :param verbal: set True to have more debug messages
    :param seed: seed for the random number generator
    :return: Object with the optimization data results
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
        optimizer=optimizer.get_name(),
        hyperparameters=optimizer.parameter,
        function=function,
        sampler=sampler,
        number_of_samples=number_of_samples,
        seeds=[seed + i for i in range(realizations)],
    )


def margin_of_victory(results: List[OptimizationResult]) -> pd.DataFrame:

    # Create df with all results
    df = pd.concat([calculate_mean_std(results[i])[0] for i, _ in enumerate(results)], axis=1)

    # Change columnnames
    optimizer_names = [results[i].optimizer for i, _ in enumerate(results)]
    df.columns = optimizer_names

    # Normalize
    min_max_scaler = preprocessing.MinMaxScaler()

    # Reshape to 1D array
    df_numpy = df.values  # returns a numpy array
    df_numpy_reshaped = df_numpy.reshape(-1, 1)

    x_scaled = min_max_scaler.fit_transform(df_numpy_reshaped)

    # Transform back
    x_scaled = x_scaled.reshape(df_numpy.shape)
    df = pd.DataFrame(x_scaled)
    df.columns = optimizer_names

    # Calculate margin of victory
    mov = []
    for name in optimizer_names:
        df_dropped = df.drop(name, axis=1)
        mov.append(df_dropped.min(axis=1) - df[name])

    # Create df with all MoV
    df_margin_of_victory = pd.concat(mov, axis=1)

    # Change columnnames
    df_margin_of_victory.columns = optimizer_names

    return df_margin_of_victory
