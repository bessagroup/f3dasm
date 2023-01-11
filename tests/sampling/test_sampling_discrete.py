import numpy as np
import pytest

pytestmark = pytest.mark.smoke

from f3dasm.sampling.randomuniform import RandomUniform


def test_correct_discrete_sampling_1(design4):
    seed = 42

    # Construct sampler
    random_uniform = RandomUniform(design=design4, seed=seed)

    numsamples = 5

    ground_truth_samples = np.array([[56, 5], [19, 4], [76, 5], [65, 5], [25, 5]])
    samples = random_uniform._sample_discrete(numsamples=numsamples)

    assert samples == pytest.approx(ground_truth_samples)


def test_correct_discrete_sampling_2(design5):
    seed = 42

    # Construct sampler
    random_uniform = RandomUniform(design=design5, seed=seed)

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
    samples = random_uniform._sample_discrete(numsamples=numsamples)
    assert samples == pytest.approx(ground_truth_samples)


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
