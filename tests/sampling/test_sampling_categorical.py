import numpy as np
import pytest

pytestmark = pytest.mark.smoke

from f3dasm.base.design import DesignSpace
from f3dasm.base.space import (
    CategoricalParameter,
    ContinuousParameter,
    DiscreteParameter,
)
from f3dasm.sampling.samplers import RandomUniformSampling


@pytest.fixture
def design():
    # Define the parameters
    x1 = ContinuousParameter(name="x1", lower_bound=2.4, upper_bound=10.3)
    x2 = DiscreteParameter(name="x2", lower_bound=5, upper_bound=80)
    x3 = ContinuousParameter(name="x3", lower_bound=10.0, upper_bound=380.3)
    x4 = CategoricalParameter(name="x4", categories=["test1", "test2", "test3"])
    x5 = DiscreteParameter(name="x5", lower_bound=3, upper_bound=6)
    x6 = CategoricalParameter(name="x6", categories=["material1", "material2", "material3"])

    # Create the design space
    space = [x1, x2, x3, x4, x5, x6]
    design = DesignSpace(space)
    return design


@pytest.fixture
def design2():
    # Define the parameters
    x1 = ContinuousParameter(name="x1", lower_bound=2.4, upper_bound=10.3)
    x2 = CategoricalParameter(name="x2", categories=["main"])
    x3 = ContinuousParameter(name="x3", lower_bound=10.0, upper_bound=380.3)
    x4 = CategoricalParameter(name="x4", categories=["test" + str(i) for i in range(80)])
    x5 = DiscreteParameter(name="x5", lower_bound=3, upper_bound=6)
    x6 = CategoricalParameter(name="x6", categories=["material" + str(i) for i in range(20)])

    # Create the design space
    space = [x1, x2, x3, x4, x5, x6]
    design = DesignSpace(space)
    return design


def test_correct_discrete_sampling_1(design):
    seed = 42

    # Construct sampler
    random_uniform = RandomUniformSampling(doe=design, seed=seed)

    numsamples = 5

    ground_truth_samples = np.array(
        [
            ["test3", "material1"],
            ["test1", "material3"],
            ["test3", "material2"],
            ["test3", "material3"],
            ["test1", "material3"],
        ]
    )
    samples = random_uniform._sample_categorical(numsamples=numsamples)

    assert (samples == ground_truth_samples).all()


def test_correct_discrete_sampling_2(design2):
    seed = 42

    # Construct sampler
    random_uniform = RandomUniformSampling(doe=design2, seed=seed)

    numsamples = 5

    ground_truth_samples = np.array(
        [
            ["main", "test51", "material6"],
            ["main", "test14", "material18"],
            ["main", "test71", "material10"],
            ["main", "test60", "material10"],
            ["main", "test20", "material3"],
        ]
    )
    samples = random_uniform._sample_categorical(numsamples=numsamples)

    assert (samples == ground_truth_samples).all()


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
