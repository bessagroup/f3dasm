from itertools import product

import numpy as np
import pandas as pd
import pytest

from f3dasm import ExperimentData
from f3dasm._src.design.samplers import Grid
from f3dasm.design import Domain

pytestmark = pytest.mark.smoke


@pytest.fixture
def sample_domain() -> Domain:
    """Fixture to provide a sample domain."""

    domain = Domain()

    domain.add_float(name='x1', low=0, high=1)
    domain.add_float(name='x2', low=2, high=4)
    domain.add_int(name='d1', low=1, high=3)
    domain.add_category(name='cat1', categories=["A", "B", "C"])
    domain.add_constant(name='const1', value=42)
    return domain


@pytest.fixture
def sample_domain_no_continuous() -> Domain:
    """Fixture to provide a sample domain."""

    domain = Domain()
    domain.add_int(name='d1', low=1, high=3)
    domain.add_category(name='cat1', categories=["A", "B", "C"])
    domain.add_constant(name='const1', value=42)
    return domain


def test_grid_sample_with_default_steps(sample_domain):
    """Test Grid sampler with default step size."""
    grid_sampler = Grid()
    experiment_data = ExperimentData(domain=sample_domain)
    grid_sampler.arm(experiment_data)

    stepsize = 0.5
    samples = grid_sampler.call(stepsize_continuous_parameters=stepsize)
    df, _ = samples.to_pandas()
    # Expected continuous values
    x1_values = np.arange(0, 1, stepsize)
    x2_values = np.arange(2, 4, stepsize)

    # Expected discrete values
    d1_values = [1, 2, 3]

    # Expected categorical values
    cat1_values = ["A", "B", "C"]

    # Generate all combinations
    expected_combinations = list(
        product(x1_values, x2_values, d1_values, cat1_values, [42]))

    # Convert to DataFrame
    expected_df = pd.DataFrame(
        expected_combinations,
        columns=["x1", "x2", "d1", "cat1", "const1"],
    )

    df.sort_values(by=['x1', 'x2', 'd1', 'cat1', 'const1'],
                   inplace=True, ignore_index=True)
    expected_df.sort_values(
        ["x1", "x2", "d1", "cat1", "const1"], inplace=True, ignore_index=True)

    # Assert equality
    pd.testing.assert_frame_equal(df, expected_df, check_dtype=False)


def test_grid_sample_with_custom_steps(sample_domain):
    """Test Grid sampler with custom step sizes for continuous parameters."""
    grid_sampler = Grid()
    experiment_data = ExperimentData(domain=sample_domain)
    grid_sampler.arm(experiment_data)

    custom_steps = {"x1": 0.25, "x2": 0.5}
    samples = grid_sampler.call(stepsize_continuous_parameters=custom_steps)
    df, _ = samples.to_pandas()

    # Expected continuous values
    x1_values = np.arange(0, 1, custom_steps["x1"])
    x2_values = np.arange(2, 4, custom_steps["x2"])

    # Expected discrete values
    d1_values = [1, 2, 3]

    # Expected categorical values
    cat1_values = ["A", "B", "C"]

    # Generate all combinations
    expected_combinations = list(
        product(x1_values, x2_values, d1_values, cat1_values, [42]))

    # Convert to DataFrame
    expected_df = pd.DataFrame(
        expected_combinations,
        columns=["x1", "x2", "d1", "cat1", "const1"],
    )

    df.sort_values(by=['x1', 'x2', 'd1', 'cat1', 'const1'],
                   inplace=True, ignore_index=True)
    expected_df.sort_values(
        ["x1", "x2", "d1", "cat1", "const1"], inplace=True, ignore_index=True)

    # Assert equality
    pd.testing.assert_frame_equal(df, expected_df, check_dtype=False)


def test_grid_sample_with_no_continuous(sample_domain_no_continuous):
    """Test Grid sampler with no continuous parameters."""
    grid_sampler = Grid()
    experiment_data = ExperimentData(domain=sample_domain_no_continuous)
    grid_sampler.arm(experiment_data)

    samples = grid_sampler.call(stepsize_continuous_parameters=None)
    df, _ = samples.to_pandas()

    # Expected discrete values
    d1_values = [1, 2, 3]

    # Expected categorical values
    cat1_values = ["A", "B", "C"]

    # Generate all combinations
    expected_combinations = list(product(d1_values, cat1_values, [42]))

    # Convert to DataFrame
    expected_df = pd.DataFrame(
        expected_combinations,
        columns=["d1", "cat1", "const1"],
    )

    df.sort_values(by=['d1', 'cat1', 'const1'],
                   inplace=True, ignore_index=True)
    expected_df.sort_values(
        ["d1", "cat1", "const1"], inplace=True, ignore_index=True)

    # Assert equality
    pd.testing.assert_frame_equal(df, expected_df, check_dtype=False)
