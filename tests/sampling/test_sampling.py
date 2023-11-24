import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from f3dasm import ExperimentData
from f3dasm.design import Domain, _ContinuousParameter

pytestmark = pytest.mark.smoke


def test_sampling_interface_not_implemented_error():
    seed = 42

    # Define the parameters
    x1 = _ContinuousParameter(lower_bound=2.4, upper_bound=10.3)
    space = {'x1': x1}

    design = Domain(space)
    with pytest.raises(KeyError):
        samples = ExperimentData(domain=design)
        samples.sample(sampler='test', n_samples=5, seed=seed)


def test_correct_sampling_ran(design3: Domain):
    seed = 42
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

    df_ground_truth = pd.DataFrame(data=ground_truth_samples)
    df_ground_truth = df_ground_truth.astype(
        {
            0: "float",
            1: "int",
            2: "float",
            3: "object",
            4: "float",
        }
    )

    samples = ExperimentData(domain=design3)
    samples.sample(sampler='random', n_samples=numsamples, seed=seed)

    samples._input_data.round(6)

    df_input, _ = samples.to_pandas()
    df_input.columns = df_ground_truth.columns

    assert_frame_equal(df_input, df_ground_truth,
                       check_dtype=False, check_exact=False, rtol=1e-6)


def test_correct_sampling_sobol(design3: Domain):
    seed = 42
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

    df_ground_truth = pd.DataFrame(data=ground_truth_samples)
    df_ground_truth = df_ground_truth.astype(
        {
            0: "float",
            1: "int",
            2: "float",
            3: "object",
            4: "float",
        }
    )

    samples = ExperimentData(domain=design3)
    samples.sample(sampler='sobol', n_samples=numsamples, seed=seed)

    df_input, _ = samples.to_pandas()
    df_input.columns = df_ground_truth.columns

    assert_frame_equal(df_input, df_ground_truth, check_dtype=False)


def test_correct_sampling_lhs(design3: Domain):
    seed = 42
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

    df_ground_truth = pd.DataFrame(data=ground_truth_samples)
    df_ground_truth = df_ground_truth.astype(
        {
            0: "float",
            1: "int",
            2: "float",
            3: "object",
            4: "float",
        }
    )

    samples = ExperimentData(domain=design3)
    samples.sample(sampler='latin', n_samples=numsamples, seed=seed)

    df_input, _ = samples.to_pandas()
    df_input.columns = df_ground_truth.columns

    assert_frame_equal(df_input, df_ground_truth, check_dtype=False)


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
