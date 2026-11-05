import warnings
from itertools import product

import numpy as np
import pandas as pd
import pytest

from f3dasm import ExperimentData
from f3dasm._src.samplers import Grid
from f3dasm.design import Domain

pytestmark = pytest.mark.smoke


@pytest.fixture
def sample_domain() -> Domain:
    """Fixture to provide a sample domain."""

    domain = Domain()

    domain.add_float(name="x1", low=0, high=1)
    domain.add_float(name="x2", low=2, high=4)
    domain.add_int(name="d1", low=1, high=3)
    domain.add_category(name="cat1", categories=["A", "B", "C"])
    domain.add_constant(name="const1", value=42)
    return domain


@pytest.fixture
def sample_domain_no_continuous() -> Domain:
    """Fixture to provide a sample domain."""

    domain = Domain()
    domain.add_int(name="d1", low=1, high=3)
    domain.add_category(name="cat1", categories=["A", "B", "C"])
    domain.add_constant(name="const1", value=42)
    return domain


def test_grid_sample_with_default_steps(sample_domain):
    """Test Grid sampler with default step size."""
    stepsize = 0.5
    grid_sampler = Grid(stepsize_continuous_parameters=stepsize)
    experiment_data = ExperimentData(domain=sample_domain)
    grid_sampler.arm(experiment_data)

    samples = grid_sampler.call(data=experiment_data)
    df, _ = samples.to_pandas()
    # Expected continuous values
    # Both bounds are part of the grid (issue #357)
    x1_values = [0.0, 0.5, 1.0]
    x2_values = [2.0, 2.5, 3.0, 3.5, 4.0]

    # Expected discrete values
    d1_values = [1, 2, 3]

    # Expected categorical values
    cat1_values = ["A", "B", "C"]

    # Generate all combinations
    expected_combinations = list(
        product(x1_values, x2_values, d1_values, cat1_values, [42])
    )

    # Convert to DataFrame
    expected_df = pd.DataFrame(
        expected_combinations,
        columns=["x1", "x2", "d1", "cat1", "const1"],
    )

    df.sort_values(
        by=["x1", "x2", "d1", "cat1", "const1"],
        inplace=True,
        ignore_index=True,
    )
    expected_df.sort_values(
        ["x1", "x2", "d1", "cat1", "const1"], inplace=True, ignore_index=True
    )

    # Assert equality
    pd.testing.assert_frame_equal(df, expected_df, check_dtype=False)


def test_grid_sample_with_custom_steps(sample_domain):
    """Test Grid sampler with custom step sizes for continuous parameters."""
    custom_steps = {"x1": 0.25, "x2": 0.5}
    grid_sampler = Grid(stepsize_continuous_parameters=custom_steps)
    experiment_data = ExperimentData(domain=sample_domain)
    grid_sampler.arm(experiment_data)

    samples = grid_sampler.call(data=experiment_data)
    df, _ = samples.to_pandas()

    # Expected continuous values
    # Both bounds are part of the grid (issue #357)
    x1_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    x2_values = [2.0, 2.5, 3.0, 3.5, 4.0]

    # Expected discrete values
    d1_values = [1, 2, 3]

    # Expected categorical values
    cat1_values = ["A", "B", "C"]

    # Generate all combinations
    expected_combinations = list(
        product(x1_values, x2_values, d1_values, cat1_values, [42])
    )

    # Convert to DataFrame
    expected_df = pd.DataFrame(
        expected_combinations,
        columns=["x1", "x2", "d1", "cat1", "const1"],
    )

    df.sort_values(
        by=["x1", "x2", "d1", "cat1", "const1"],
        inplace=True,
        ignore_index=True,
    )
    expected_df.sort_values(
        ["x1", "x2", "d1", "cat1", "const1"], inplace=True, ignore_index=True
    )

    # Assert equality
    pd.testing.assert_frame_equal(df, expected_df, check_dtype=False)


def test_grid_sample_with_no_continuous(sample_domain_no_continuous):
    """Test Grid sampler with no continuous parameters."""
    grid_sampler = Grid()
    experiment_data = ExperimentData(domain=sample_domain_no_continuous)
    grid_sampler.arm(experiment_data)

    samples = grid_sampler.call(data=experiment_data)
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

    df.sort_values(
        by=["d1", "cat1", "const1"], inplace=True, ignore_index=True
    )
    expected_df.sort_values(
        ["d1", "cat1", "const1"], inplace=True, ignore_index=True
    )

    # Assert equality
    pd.testing.assert_frame_equal(df, expected_df, check_dtype=False)


def test_grid_sample_continuous_without_stepsize_raises():
    """Mixed categorical + continuous domain with no stepsize must raise.

    Regression for issue #318: the grid sampler used to silently return a
    degenerate one-row grid (the categorical cross-product times an empty
    continuous contribution) when the user forgot to pass
    `stepsize_continuous_parameters` for a domain that contains continuous
    parameters. It now raises a clear ValueError instead.
    """
    domain = Domain()
    domain.add_category(name="ad_mode", categories=["unroll"])
    domain.add_float(name="target_pt_a1", low=0.25, high=2.0)
    domain.add_float(name="target_pt_a2", low=0.25, high=2.0)

    grid_sampler = Grid()  # no stepsize_continuous_parameters
    experiment_data = ExperimentData(domain=domain)
    grid_sampler.arm(experiment_data)

    with pytest.raises(ValueError, match="stepsize_continuous_parameters"):
        grid_sampler.call(data=experiment_data)


# ======================= Continuous grid bounds (issue #357) ===============


def _grid_values(lower, upper, stepsize):
    """Return the sorted grid values of a single continuous parameter."""
    domain = Domain()
    domain.add_float(name="x", low=lower, high=upper)
    samples = Grid(stepsize_continuous_parameters=stepsize).call(
        data=ExperimentData(domain=domain)
    )
    df, _ = samples.to_pandas()
    return np.sort(df["x"].to_numpy(dtype=float))


@pytest.mark.parametrize(
    ("lower", "upper", "stepsize", "expected"),
    [
        (0.0, 1.0, 0.25, [0.0, 0.25, 0.5, 0.75, 1.0]),
        (0.0, 1.0, 0.5, [0.0, 0.5, 1.0]),
        (1.0, 4.0, 0.5, [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]),
        (1.0, 1.3, 0.1, [1.0, 1.1, 1.2, 1.3]),
        (0.0, 3.0, 1, [0.0, 1.0, 2.0, 3.0]),
    ],
)
def test_grid_includes_both_bounds(lower, upper, stepsize, expected):
    """Steps that divide the range give a grid from bound to bound."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        values = _grid_values(lower, upper, stepsize)
    np.testing.assert_allclose(values, expected)
    assert values[0] == lower
    assert values[-1] == upper


def test_grid_dict_stepsize_includes_both_bounds():
    """The per-parameter stepsize form behaves like the scalar form."""
    values = _grid_values(0.0, 1.0, {"x": 0.25})
    np.testing.assert_allclose(values, [0.0, 0.25, 0.5, 0.75, 1.0])


@pytest.mark.parametrize(
    ("stepsize", "expected"),
    [
        (0.3, [0.0, 1 / 3, 2 / 3, 1.0]),  # 3.33 steps -> 3
        (0.4, [0.0, 1 / 3, 2 / 3, 1.0]),  # 2.5 steps -> 3 (ties round up)
        (0.45, [0.0, 0.5, 1.0]),  # 2.22 steps -> 2
        (5.0, [0.0, 1.0]),  # larger than the range: the two bounds
    ],
)
def test_grid_rounds_step_that_does_not_divide_range(stepsize, expected):
    """A step that does not divide the range is rounded to one that does."""
    with pytest.warns(UserWarning, match="does not divide"):
        values = _grid_values(0.0, 1.0, stepsize)
    np.testing.assert_allclose(values, expected)


def test_grid_two_level_corners():
    """A step equal to the range gives the corners of the domain."""
    domain = Domain()
    domain.add_float(name="x1", low=0.0, high=1.0)
    domain.add_float(name="x2", low=0.0, high=1.0)
    samples = Grid(stepsize_continuous_parameters=1.0).call(
        data=ExperimentData(domain=domain)
    )
    df, _ = samples.to_pandas()
    corners = {tuple(row) for row in df[["x1", "x2"]].to_numpy(dtype=float)}
    assert corners == {(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)}


@pytest.mark.parametrize("stepsize", [0.0, -0.5, {"x": 0.0}, {"x": -0.5}])
def test_grid_non_positive_step_raises(stepsize):
    with pytest.raises(ValueError, match="larger than 0"):
        _grid_values(0.0, 1.0, stepsize)
