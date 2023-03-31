import numpy as np
import pandas as pd
import pytest

from f3dasm.design.design import DesignSpace
from f3dasm.design.parameter import ContinuousParameter
from f3dasm.sampling.latinhypercube import LatinHypercube
from f3dasm.sampling.randomuniform import RandomUniform
from f3dasm.sampling.sampler import Sampler
from f3dasm.sampling.sobolsequence import SobolSequence

pytestmark = pytest.mark.smoke


# Sampling interface


def test_sampling_interface_not_implemented_error():
    seed = 42

    class NewSamplingStrategy(Sampler):
        pass

    # Define the parameters
    x1 = ContinuousParameter(name="x1", lower_bound=2.4, upper_bound=10.3)
    space = [x1]

    design = DesignSpace(space)
    new_sampler = NewSamplingStrategy(design=design, seed=seed)
    with pytest.raises(NotImplementedError):
        _ = new_sampler.sample_continuous(numsamples=5)


def test_correct_sampling_ran(design3: DesignSpace):
    seed = 42
    # Construct sampler
    random_sequencing = RandomUniform(design=design3, seed=seed)

    numsamples = 5

    ground_truth_samples = np.array(
        [
            [5.358867, 25, 362.049508, "test1", 5.504359],
            [7.129402, 37, 67.773703, "test1", 1.645163],
            [2.858861, 80, 330.745027, "test1", 4.627471],
            [7.993773, 62, 17.622438, "test3", 7.098396],
            [8.976297, 26, 88.629173, "test3", 1.818227],
        ]
    )

    columnnames = [
        ["input"] * design3.get_number_of_input_parameters(),
        ["x1", "x2", "x3", "x4", "x5"],
    ]
    df_ground_truth = pd.DataFrame(data=ground_truth_samples, columns=columnnames)
    df_ground_truth = df_ground_truth.astype(
        {
            ("input", "x1"): "float",
            ("input", "x2"): "int",
            ("input", "x3"): "float",
            ("input", "x4"): "category",
            ("input", "x5"): "float",
        }
    )

    samples = random_sequencing.get_samples(numsamples=numsamples)
    samples = samples.data.round(6)

    assert df_ground_truth.equals(samples)


def test_correct_sampling_sobol(design3: DesignSpace):
    seed = 42

    # Construct sampler
    sobol_sequencing = SobolSequence(design=design3, seed=seed)

    numsamples = 5

    ground_truth_samples = np.array(
        [
            [2.4000, 56, 10.0000, "test3", 0.6000],
            [6.3500, 19, 195.1500, "test2", 3.9500],
            [8.3250, 76, 102.5750, "test3", 2.2750],
            [4.3750, 65, 287.7250, "test3", 5.6250],
            [5.3625, 25, 148.8625, "test3", 4.7875],
        ]
    )

    columnnames = [
        ["input"] * design3.get_number_of_input_parameters(),
        ["x1", "x2", "x3", "x4", "x5"],
    ]
    df_ground_truth = pd.DataFrame(data=ground_truth_samples, columns=columnnames)
    df_ground_truth = df_ground_truth.astype(
        {
            ("input", "x1"): "float",
            ("input", "x2"): "int",
            ("input", "x3"): "float",
            ("input", "x4"): "category",
            ("input", "x5"): "float",
        }
    )

    samples = sobol_sequencing.get_samples(numsamples=numsamples)
    samples = samples.data.round(6)
    assert df_ground_truth.equals(samples)


def test_correct_sampling_lhs(design3: DesignSpace):
    seed = 42

    # Construct sampler
    lhs_sampler = LatinHypercube(design=design3, seed=seed)

    numsamples = 5

    ground_truth_samples = np.array(
        [
            [8.258755, 64, 95.614741, "test2", 1.580872],
            [5.651772, 19, 222.269005, "test3", 2.149033],
            [4.925880, 66, 80.409902, "test3", 5.919679],
            [2.991773, 66, 233.704488, "test1", 6.203645],
            [10.035259, 51, 321.965835, "test3", 4.085494],
        ]
    )

    columnnames = [
        ["input"] * design3.get_number_of_input_parameters(),
        ["x1", "x2", "x3", "x4", "x5"],
    ]
    df_ground_truth = pd.DataFrame(data=ground_truth_samples, columns=columnnames)
    df_ground_truth = df_ground_truth.astype(
        {
            ("input", "x1"): "float",
            ("input", "x2"): "int",
            ("input", "x3"): "float",
            ("input", "x4"): "category",
            ("input", "x5"): "float",
        }
    )

    samples = lhs_sampler.get_samples(numsamples=numsamples)
    samples = samples.data.round(6)

    assert df_ground_truth.equals(samples)


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
