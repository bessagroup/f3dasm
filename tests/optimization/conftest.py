import pytest

from f3dasm.design import (CategoricalParameter, ContinuousParameter,
                           DiscreteParameter, Domain)
from f3dasm.sampling import RandomUniform


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

    # Create the design space
    design = Domain(space=input_parameters)

    # Set the lower_bound and upper_bound of 'y' to None, indicating it has no bounds

    random_sampler = RandomUniform(domain=design, seed=seed)
    data = random_sampler.get_samples(numsamples=N)

    return data
