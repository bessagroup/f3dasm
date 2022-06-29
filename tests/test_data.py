import pytest
from f3dasm.src.data import Data
from f3dasm.src.designofexperiments import DoE
from f3dasm.src.space import CategoricalSpace, ContinuousSpace, DiscreteSpace

from f3dasm.sampling.randomuniform import RandomUniform


@pytest.fixture
def data():
    seed = 42
    N = 10  # Number of samples

    # Define the parameters
    x1 = ContinuousSpace(name="x1", lower_bound=2.4, upper_bound=10.3)
    x2 = DiscreteSpace(name="x2", lower_bound=5, upper_bound=80)
    x3 = ContinuousSpace(name="x3", lower_bound=10.0, upper_bound=380.3)
    x4 = CategoricalSpace(name="x4", categories=["test1", "test2", "test3"])
    x5 = ContinuousSpace(name="x5", lower_bound=0.6, upper_bound=7.3)

    y1 = ContinuousSpace(name="y1", lower_bound=0.6, upper_bound=7.3)
    y2 = ContinuousSpace(name="y2", lower_bound=6.0, upper_bound=20.0)
    # Create the design space
    input_space = [x1, x2, x3, x4, x5]
    output_space = [y1, y2]
    design = DoE(input_space=input_space, output_space=output_space)

    # Create Data object
    data = Data(doe=design)

    random_sampler = RandomUniform(doe=design, seed=seed)
    samples = random_sampler.get_samples(numsamples=N)

    data.add(samples)

    return data


def test_get_output_data(data: Data):
    truth = data.data[[("output", "y1"), ("output", "y2")]]["output"]

    assert all(data.get_output_data() == truth)


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
