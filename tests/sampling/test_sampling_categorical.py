import numpy as np
import pytest

from f3dasm.sampling.randomuniform import RandomUniform

pytestmark = pytest.mark.smoke


def test_correct_discrete_sampling_1(design):
    seed = 42

    # Construct sampler
    random_uniform = RandomUniform(design=design, seed=seed)

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
    random_uniform = RandomUniform(design=design2, seed=seed)

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
