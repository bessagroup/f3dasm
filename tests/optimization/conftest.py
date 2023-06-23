import pytest

from f3dasm.design.design import DesignSpace
from f3dasm.design.parameter import (CategoricalParameter, ContinuousParameter,
                                     DiscreteParameter)
from f3dasm.sampling.randomuniform import RandomUniform


@pytest.fixture(scope="package")
def data():
    seed = 42
    N = 50  # Number of samples

    # Define the parameters
    input_parameters = {
        "x1": ContinuousParameter(lower_bound=2.4, upper_bound=10.3),
        "x2": ContinuousParameter(lower_bound=10.0, upper_bound=380.3),
        "x3": ContinuousParameter(lower_bound=0.6, upper_bound=7.3),
    }

    output_parameters = {
        "y": ContinuousParameter()}

    # Create the design space
    design = DesignSpace(input_space=input_parameters, output_space=output_parameters)

    # Set the lower_bound and upper_bound of 'y' to None, indicating it has no bounds

    random_sampler = RandomUniform(design=design, seed=seed)
    data = random_sampler.get_samples(numsamples=N)

    return data
