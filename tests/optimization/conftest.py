import pytest

from f3dasm import ExperimentData
from f3dasm._src.design.parameter import _ContinuousParameter
from f3dasm.design import Domain


@pytest.fixture(scope="package")
def data():
    seed = 42
    N = 50  # Number of samples

    # Define the parameters
    input_parameters = {
        "x1": _ContinuousParameter(lower_bound=2.4, upper_bound=10.3),
        "x2": _ContinuousParameter(lower_bound=10.0, upper_bound=380.3),
        "x3": _ContinuousParameter(lower_bound=0.6, upper_bound=7.3),
    }

    # Create the design space
    design = Domain(space=input_parameters)

    # Set the lower_bound and upper_bound of 'y' to None, indicating it has no bounds
    return ExperimentData.from_sampling(sampler='random', domain=design, n_samples=N, seed=seed)
