from typing import List, Type
import numpy as np
from f3dasm.base.data import Data

from f3dasm.base.optimization import Optimizer
from f3dasm.base.samplingmethod import SamplingInterface
from f3dasm.base.simulation import Function


def run_multiple_realizations(
    algorithm: Type[Optimizer],
    function: Function,
    sampler: SamplingInterface,
    iterations: int,
    realizations: int,
) -> List[Data]:
    """Run multiple realizations of the same algorithm on a benchmark function"""

    # Create a random seed
    seed = np.random.randint(low=0, high=1e5)
    all_data = []

    for _ in range(realizations):

        # Set function seed
        function.set_seed(seed)

        # Sample
        sampler.set_seed(seed)
        samples = sampler.get_samples(numsamples=30)
        # TODO: make this numsamples a hyperparameter of the algorithm?

        samples.add_output(output=function.eval(samples), label="y")

        # Construct optimizer object
        optimizer = algorithm(
            data=samples, seed=seed, population=samples.get_number_of_datapoints()
        )

        # Iterate
        optimizer.iterate(iterations=iterations, function=function)

        # Reset the parameters
        optimizer.init_parameters()

        all_data.append(optimizer.extract_data())

        # Increase seed
        seed += 1

    return all_data
