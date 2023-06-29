import pytest

from f3dasm.design.design import DesignSpace
from f3dasm.design.parameter import (CategoricalParameter, ContinuousParameter,
                                     DiscreteParameter)
from f3dasm.sampling.randomuniform import RandomUniform


@pytest.fixture(scope="package")
def data():
    seed = 42
    N = 10  # Number of samples

    # Define the parameters
    x1 = ContinuousParameter(lower_bound=2.4, upper_bound=10.3)
    x2 = DiscreteParameter(lower_bound=5, upper_bound=80)
    x3 = ContinuousParameter(lower_bound=10.0, upper_bound=380.3)
    x4 = CategoricalParameter(categories=["test1", "test2", "test3"])
    x5 = ContinuousParameter(lower_bound=0.6, upper_bound=7.3)

    y1 = ContinuousParameter(lower_bound=0.6, upper_bound=7.3)
    y2 = ContinuousParameter(lower_bound=6.0, upper_bound=20.0)
    # Create the design space
    input_space = {'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5}
    output_space = {'y1': y1, 'y2': y2}
    design = DesignSpace(input_space=input_space, output_space=output_space)

    random_sampler = RandomUniform(design=design, seed=seed)
    data = random_sampler.get_samples(numsamples=N)

    return data
