import pytest

pytestmark = pytest.mark.smoke

from f3dasm.base.data import Data
from f3dasm.base.design import DesignSpace
from f3dasm.base.space import (
    CategoricalParameter,
    ContinuousParameter,
    DiscreteParameter,
)
from f3dasm.sampling.randomuniform import RandomUniform


@pytest.fixture
def data():
    seed = 42
    N = 10  # Number of samples

    # Define the parameters
    x1 = ContinuousParameter(name="x1", lower_bound=2.4, upper_bound=10.3)
    x2 = DiscreteParameter(name="x2", lower_bound=5, upper_bound=80)
    x3 = ContinuousParameter(name="x3", lower_bound=10.0, upper_bound=380.3)
    x4 = CategoricalParameter(name="x4", categories=["test1", "test2", "test3"])
    x5 = ContinuousParameter(name="x5", lower_bound=0.6, upper_bound=7.3)

    y1 = ContinuousParameter(name="y1", lower_bound=0.6, upper_bound=7.3)
    y2 = ContinuousParameter(name="y2", lower_bound=6.0, upper_bound=20.0)
    # Create the design space
    input_space = [x1, x2, x3, x4, x5]
    output_space = [y1, y2]
    design = DesignSpace(input_space=input_space, output_space=output_space)

    random_sampler = RandomUniform(design=design, seed=seed)
    data = random_sampler.get_samples(numsamples=N)

    return data


def test_get_output_data(data: Data):
    truth = data.data[[("output", "y1"), ("output", "y2")]]["output"]

    assert all(data.get_output_data() == truth)


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
