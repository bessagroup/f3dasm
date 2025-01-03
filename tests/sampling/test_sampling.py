import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from f3dasm import ExperimentData
from f3dasm._src.design.parameter import ContinuousParameter
from f3dasm.design import Domain

pytestmark = pytest.mark.smoke


def test_sampling_interface_not_implemented_error():
    seed = 42

    # Define the parameters
    x1 = ContinuousParameter(lower_bound=2.4, upper_bound=10.3)
    space = {'x1': x1}

    design = Domain(space)
    with pytest.raises(KeyError):
        samples = ExperimentData(domain=design)
        samples.sample(sampler='test', n_samples=5, seed=seed)


def test_correct_sampling_ran(design3: Domain):
    seed = 42
    numsamples = 5

    ground_truth_samples = np.array([
        [8.514253, 11, 172.516686, 'test1', 6.352606],
        [7.909207, 63, 44.873872, 'test3', 7.13667],
        [8.413004, 54, 301.079612, 'test2', 1.458361],
        [5.958049, 38, 147.306508, 'test2', 6.809325],
        [7.486534, 37, 314.668625, 'test2', 3.570875]
    ])

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

    df_input, _ = samples.to_pandas()
    df_input.columns = df_ground_truth.columns

    assert_frame_equal(df_input, df_ground_truth,
                       check_dtype=False, check_exact=False, rtol=1e-6)


def test_correct_sampling_sobol(design3: Domain):
    seed = 42
    numsamples = 5

    ground_truth_samples = np.array(
        [
            [2.4000, 11, 10.0000, "test1", 0.6000],
            [6.3500, 63, 195.1500, "test3", 3.9500],
            [8.3250, 54, 102.5750, "test2", 2.2750],
            [4.3750, 38, 287.7250, "test2", 5.6250],
            [5.3625, 37, 148.8625, "test2", 4.7875],
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
            [8.258755, 11, 95.614741, "test1", 1.580872],
            [5.651772, 63, 222.269005, "test3", 2.149033],
            [4.925880, 54, 80.409902, "test2", 5.919679],
            [2.991773, 38, 233.704488, "test2", 6.203645],
            [10.035259, 37, 321.965835, "test2", 4.085494],
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
