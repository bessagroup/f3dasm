import numpy as np
import pytest
from f3dasm.sampling.randomuniform import RandomUniform
from f3dasm.src.designofexperiments import DoE
from f3dasm.src.space import CategoricalSpace, ContinuousSpace, DiscreteSpace


@pytest.fixture
def doe1():
    # Define the parameters
    x1 = ContinuousSpace(name="x1", lower_bound=2.4, upper_bound=10.3)
    x2 = DiscreteSpace(name="x2", lower_bound=5, upper_bound=80)
    x3 = ContinuousSpace(name="x3", lower_bound=10.0, upper_bound=380.3)
    x4 = CategoricalSpace(name="x4", categories=["test1", "test2", "test3"])
    x5 = DiscreteSpace(name="x5", lower_bound=3, upper_bound=6)

    # Create the design space
    space = [x1, x2, x3, x4, x5]
    design = DoE(space)
    return design


@pytest.fixture
def doe2():
    # Define the parameters
    x1 = ContinuousSpace(name="x1", lower_bound=2.4, upper_bound=10.3)
    x2 = DiscreteSpace(name="x2", lower_bound=5, upper_bound=80)
    x3 = ContinuousSpace(name="x3", lower_bound=10.0, upper_bound=380.3)
    x4 = CategoricalSpace(name="x4", categories=["test1", "test2", "test3"])
    x5 = DiscreteSpace(name="x5", lower_bound=3, upper_bound=6)
    x6 = DiscreteSpace(name="x6", lower_bound=500, upper_bound=532)

    # Create the design space
    space = [x1, x2, x3, x4, x5, x6]
    design = DoE(space)

    return design


def test_correct_discrete_sampling_1(doe1):
    seed = 42

    # Construct sampler
    random_uniform = RandomUniform(doe=doe1, seed=seed)

    numsamples = 5

    ground_truth_samples = np.array([[56, 5], [19, 4], [76, 5], [65, 5], [25, 5]])
    samples = random_uniform.sample_discrete(numsamples=numsamples, doe=doe1)

    assert samples == pytest.approx(ground_truth_samples)


def test_correct_discrete_sampling_2(doe2):
    seed = 42

    # Construct sampler
    random_uniform = RandomUniform(doe=doe2, seed=seed)

    numsamples = 5

    ground_truth_samples = np.array(
        [
            [56, 5, 510],
            [19, 4, 523],
            [76, 5, 523],
            [65, 5, 502],
            [25, 5, 521],
        ]
    )
    samples = random_uniform.sample_discrete(numsamples=numsamples, doe=doe2)
    assert samples == pytest.approx(ground_truth_samples)


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
