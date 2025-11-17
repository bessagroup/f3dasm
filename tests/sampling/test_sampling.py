import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from f3dasm import ExperimentData, create_sampler
from f3dasm._src.design.parameter import ContinuousParameter
from f3dasm.design import Domain

pytestmark = pytest.mark.smoke


def test_sampling_interface_not_implemented_error():
    seed = 42

    # Define the parameters
    x1 = ContinuousParameter(lower_bound=2.4, upper_bound=10.3)
    space = {"x1": x1}

    design = Domain(space)
    samples = ExperimentData(domain=design)

    with pytest.raises(KeyError):
        sampler = create_sampler(sampler="test", seed=seed)
        _ = sampler.call(data=samples, n_samples=5)


def test_correct_sampling_ran(design3: Domain):
    seed = 42
    numsamples = 5

    ground_truth_samples = np.array(
        [
            [8.514253, 11, 172.516686, "test1", 6.352606],
            [7.909207, 63, 44.873872, "test3", 7.13667],
            [8.413004, 54, 301.079612, "test2", 1.458361],
            [5.958049, 38, 147.306508, "test2", 6.809325],
            [7.486534, 37, 314.668625, "test2", 3.570875],
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

    sampler = create_sampler(sampler="random", seed=seed)
    samples = sampler.call(data=samples, n_samples=numsamples)

    df_input, _ = samples.to_pandas()
    df_input = df_input.reindex(sorted(df_input.columns), axis=1)
    df_input.columns = df_ground_truth.columns

    assert_frame_equal(
        df_input,
        df_ground_truth,
        check_dtype=False,
        check_exact=False,
        rtol=1e-6,
    )


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
    sampler = create_sampler(sampler="sobol", seed=seed)

    samples = sampler.call(data=samples, n_samples=numsamples)

    df_input, _ = samples.to_pandas()
    df_input = df_input.reindex(sorted(df_input.columns), axis=1)
    df_input.columns = df_ground_truth.columns

    assert_frame_equal(df_input, df_ground_truth, check_dtype=False)


def test_correct_sampling_lhs(design3: Domain):
    seed = 42
    numsamples = 5

    ground_truth_samples = np.array(
        [
            [3.622851, 11, 91.034774, "test1", 6.554175],
            [5.081841, 63, 367.173725, "test3", 5.861865],
            [7.851610, 54, 42.503337, "test2", 3.451672],
            [9.737307, 38, 216.335922, "test2", 3.247334],
            [6.762601, 37, 259.641302, "test2", 1.750521],
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

    sampler = create_sampler(sampler="latin", seed=seed)

    samples = sampler.call(data=samples, n_samples=numsamples)

    df_input, _ = samples.to_pandas()
    df_input = df_input.reindex(sorted(df_input.columns), axis=1)
    df_input.columns = df_ground_truth.columns

    assert_frame_equal(df_input, df_ground_truth, check_dtype=False)


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
