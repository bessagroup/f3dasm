import numpy as np
import pytest
from f3dasm.sampling.randomuniform import RandomUniform
from f3dasm.src.designofexperiments import DoE
from f3dasm.src.space import CategoricalSpace, ContinuousSpace, DiscreteSpace


def test_correct_discrete_sampling_1():
    seed = 42

    # Define the parameters
    x1 = ContinuousSpace(name="x1", lower_bound=2.4, upper_bound=10.3)
    x2 = DiscreteSpace(name="x2", lower_bound=5, upper_bound=80)
    x3 = ContinuousSpace(name="x3", lower_bound=10.0, upper_bound=380.3)
    x4 = CategoricalSpace(name="x4", categories=["test1", "test2", "test3"])
    x5 = DiscreteSpace(name="x5", lower_bound=3, upper_bound=6)
    x6 = CategoricalSpace(name="x6", categories=["material1", "material2", "material3"])

    # Create the design space
    space = [x1, x2, x3, x4, x5, x6]
    design = DoE(space)

    # Construct sampler
    random_uniform = RandomUniform(doe=design, seed=seed)

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
    samples = random_uniform.sample_categorical(numsamples=numsamples, doe=design)

    assert (samples == ground_truth_samples).all()


def test_correct_discrete_sampling_2():
    seed = 42

    # Define the parameters
    x1 = ContinuousSpace(name="x1", lower_bound=2.4, upper_bound=10.3)
    x2 = CategoricalSpace(name="x2", categories=["main"])
    x3 = ContinuousSpace(name="x3", lower_bound=10.0, upper_bound=380.3)
    x4 = CategoricalSpace(name="x4", categories=["test" + str(i) for i in range(80)])
    x5 = DiscreteSpace(name="x5", lower_bound=3, upper_bound=6)
    x6 = CategoricalSpace(
        name="x6", categories=["material" + str(i) for i in range(20)]
    )

    # Create the design space
    space = [x1, x2, x3, x4, x5, x6]
    design = DoE(space)

    # Construct sampler
    random_uniform = RandomUniform(doe=design, seed=seed)

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
    samples = random_uniform.sample_categorical(numsamples=numsamples, doe=design)

    assert (samples == ground_truth_samples).all()


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
