import pytest

from f3dasm import ExperimentData, create_sampler
from f3dasm._src.design.parameter import ContinuousParameter
from f3dasm.design import Domain
import numpy as np

DIM = 2


@pytest.fixture(scope="package")
def data():
    domain = Domain()
    domain.add_parameter('x')
    domain.add_output('y')

    x0 = np.array([0.5]*DIM)
    return ExperimentData(
        domain=domain,
        input_data=[{'x': x0}],
    )

    # seed = 42
    # N = 50  # Number of samples

    # # Define the parameters
    # input_parameters = {
    #     "x1": ContinuousParameter(lower_bound=2.4, upper_bound=10.3),
    #     "x2": ContinuousParameter(lower_bound=10.0, upper_bound=380.3),
    #     "x3": ContinuousParameter(lower_bound=0.6, upper_bound=7.3),
    # }

    # # Create the design space
    # design = Domain(input_space=input_parameters)

    # data = ExperimentData(domain=design)

    # sampler = create_sampler(sampler='random', seed=seed)
    # sampler.arm(data=data)

    # return sampler.call(data=data, n_samples=N)
