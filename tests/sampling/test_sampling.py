import numpy as np
import pandas as pd
import pytest
from f3dasm.sampling.latinhypercube import LatinHypercube
from f3dasm.sampling.randomuniform import RandomUniform
from f3dasm.sampling.sobolsequence import SobolSequencing
from f3dasm.src.designofexperiments import DoE
from f3dasm.src.space import CategoricalSpace, ContinuousSpace, DiscreteSpace


@pytest.fixture
def design():
    # Define the parameters
    x1 = ContinuousSpace(name="x1", lower_bound=2.4, upper_bound=10.3)
    x2 = DiscreteSpace(name="x2", lower_bound=5, upper_bound=80)
    x3 = ContinuousSpace(name="x3", lower_bound=10.0, upper_bound=380.3)
    x4 = CategoricalSpace(name="x4", categories=["test1", "test2", "test3"])
    x5 = ContinuousSpace(name="x5", lower_bound=0.6, upper_bound=7.3)

    # Create the design space
    space = [x1, x2, x3, x4, x5]
    design = DoE(space)
    return design


def test_correct_sampling_ran(design):
    seed = 42
    # Construct sampler
    sobol_sequencing = RandomUniform(doe=design, seed=seed)

    numsamples = 5

    ground_truth_samples = np.array(
        [
            [5.358867, 362.049508, 5.504359, 25, "test1"],
            [7.129402, 67.773703, 1.645163, 37, "test1"],
            [2.858861, 330.745027, 4.627471, 80, "test1"],
            [7.993773, 17.622438, 7.098396, 62, "test3"],
            [8.976297, 88.629173, 1.818227, 26, "test3"],
        ]
    )

    columnnames = [
        ["input"] * design.getNumberOfInputParameters(),
        ["x1", "x3", "x5", "x2", "x4"],
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
    samples = samples.round(6)

    print(df_ground_truth)
    assert df_ground_truth.equals(samples)


def test_correct_sampling_sobol(design):
    seed = 42

    # Construct sampler
    sobol_sequencing = SobolSequencing(doe=design, seed=seed)

    numsamples = 5

    ground_truth_samples = np.array(
        [
            [2.4000, 10.0000, 0.6000, 56, "test3"],
            [6.3500, 195.1500, 3.9500, 19, "test2"],
            [8.3250, 102.5750, 2.2750, 76, "test3"],
            [4.3750, 287.7250, 5.6250, 65, "test3"],
            [5.3625, 148.8625, 4.7875, 25, "test3"],
        ]
    )

    columnnames = [
        ["input"] * design.getNumberOfInputParameters(),
        ["x1", "x3", "x5", "x2", "x4"],
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
    samples = samples.round(6)

    assert df_ground_truth.equals(samples)


def test_correct_sampling_lhs(design):
    seed = 42

    # Construct sampler
    sobol_sequencing = LatinHypercube(doe=design, seed=seed)

    numsamples = 5

    ground_truth_samples = np.array(
        [
            [8.258755, 95.614741, 1.580872, 64, "test2"],
            [5.651772, 222.269005, 2.149033, 19, "test3"],
            [4.925880, 80.409902, 5.919679, 66, "test3"],
            [2.991773, 233.704488, 6.203645, 66, "test1"],
            [10.035259, 321.965835, 4.085494, 51, "test3"],
        ]
    )

    columnnames = [
        ["input"] * design.getNumberOfInputParameters(),
        ["x1", "x3", "x5", "x2", "x4"],
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
    samples = samples.round(6)

    assert df_ground_truth.equals(samples)


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
