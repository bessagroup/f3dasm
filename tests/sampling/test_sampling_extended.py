"""Extended tests for sampling methods."""

import numpy as np
import pytest

from f3dasm import ExperimentData, create_sampler
from f3dasm._src.design.parameter import (
    ArrayParameter,
    CategoricalParameter,
    ConstantParameter,
    ContinuousParameter,
    DiscreteParameter,
)
from f3dasm._src.samplers import (
    Grid,
    grid_values_categorical_parameters,
    grid_values_constant_parameters,
    grid_values_continuous_parameters,
    grid_values_discrete_parameters,
    latin_sample_array_parameters,
    next_power_of_two,
    sobol_sample_array_parameters,
)
from f3dasm.design import Domain

pytestmark = pytest.mark.smoke


# ======================= create_sampler =======================


def test_create_sampler_invalid_name():
    with pytest.raises(KeyError):
        create_sampler(sampler="nonexistent_sampler")


def test_create_sampler_invalid_type():
    with pytest.raises(TypeError):
        create_sampler(sampler=12345)


def test_create_sampler_random():
    sampler = create_sampler(sampler="random", seed=42)
    assert sampler is not None


def test_create_sampler_latin():
    sampler = create_sampler(sampler="latin", seed=42)
    assert sampler is not None


def test_create_sampler_sobol():
    sampler = create_sampler(sampler="sobol", seed=42)
    assert sampler is not None


def test_create_sampler_grid():
    sampler = create_sampler(sampler="grid")
    assert sampler is not None


# ======================= next_power_of_two =======================


def test_next_power_of_two_zero():
    assert next_power_of_two(0) == 1


def test_next_power_of_two_one():
    assert next_power_of_two(1) == 1


def test_next_power_of_two_two():
    assert next_power_of_two(2) == 2


def test_next_power_of_two_three():
    assert next_power_of_two(3) == 4


def test_next_power_of_two_1024():
    assert next_power_of_two(1024) == 1024


def test_next_power_of_two_1025():
    assert next_power_of_two(1025) == 2048


# ======================= Random sampler properties =======================


def test_random_sampler_seed_reproducibility():
    domain = Domain()
    domain.add_float("x0", 0.0, 1.0)
    domain.add_float("x1", 0.0, 1.0)

    data1 = ExperimentData(domain=domain)
    sampler1 = create_sampler(sampler="random", seed=42)
    result1 = sampler1.call(data=data1, n_samples=10)

    data2 = ExperimentData(domain=domain)
    sampler2 = create_sampler(sampler="random", seed=42)
    result2 = sampler2.call(data=data2, n_samples=10)

    df1, _ = result1.to_pandas()
    df2, _ = result2.to_pandas()
    np.testing.assert_array_equal(df1.values, df2.values)


def test_random_sampler_respects_bounds():
    domain = Domain()
    domain.add_float("x0", -5.0, 5.0)
    domain.add_float("x1", 0.0, 100.0)

    data = ExperimentData(domain=domain)
    sampler = create_sampler(sampler="random", seed=42)
    result = sampler.call(data=data, n_samples=50)

    df, _ = result.to_pandas()
    assert df["x0"].min() >= -5.0
    assert df["x0"].max() <= 5.0
    assert df["x1"].min() >= 0.0
    assert df["x1"].max() <= 100.0


def test_latin_hypercube_sample_count():
    domain = Domain()
    domain.add_float("x0", 0.0, 1.0)
    domain.add_float("x1", 0.0, 1.0)

    data = ExperimentData(domain=domain)
    sampler = create_sampler(sampler="latin", seed=42)
    result = sampler.call(data=data, n_samples=20)
    assert len(result) == 20


def test_sobol_sample_count():
    domain = Domain()
    domain.add_float("x0", 0.0, 1.0)
    domain.add_float("x1", 0.0, 1.0)

    data = ExperimentData(domain=domain)
    sampler = create_sampler(sampler="sobol", seed=42)
    result = sampler.call(data=data, n_samples=16)
    assert len(result) == 16


# ======================= Sampler with empty domain =======================


def test_sampler_continuous_only():
    domain = Domain()
    domain.add_float("x0", 0.0, 1.0)

    data = ExperimentData(domain=domain)
    sampler = create_sampler(sampler="random", seed=42)
    result = sampler.call(data=data, n_samples=5)
    assert len(result) == 5


def test_sampler_discrete_only():
    domain = Domain()
    domain.add_int("d0", 0, 10)

    data = ExperimentData(domain=domain)
    sampler = create_sampler(sampler="random", seed=42)
    result = sampler.call(data=data, n_samples=5)
    assert len(result) == 5
    df, _ = result.to_pandas()
    # All values should be integers within bounds
    for val in df["d0"]:
        assert 0 <= int(val) <= 10


def test_sampler_discrete_large_bound_bounded_memory():
    # Regression for issue #270: the random sampler used to do
    # `rng.choice(range(low, high+1, step))`, which materializes the full
    # range into a numpy array and OOMs for high ~ 1e12. The replacement
    # uses `rng.integers` directly and must run in bounded memory while
    # still producing values inside the domain.
    import tracemalloc

    upper = int(1e12)
    domain = Domain()
    domain.add_int("x", low=0, high=upper)
    data = ExperimentData(domain=domain)
    sampler = create_sampler(sampler="random", seed=2024)

    tracemalloc.start()
    result = sampler.call(data=data, n_samples=4)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # The bug allocated O((high-low)/step) bytes; with high == 1e12 that
    # is many gigabytes. A correct implementation only needs a few KB for
    # the 4 sampled values. Set the cap generously (8 MB) so the test
    # stays portable while still proving the bug is gone.
    assert peak < 8 * 1024 * 1024
    df, _ = result.to_pandas()
    for val in df["x"]:
        assert 0 <= int(val) <= upper


def test_sampler_discrete_with_step():
    # The step != 1 code path is exercised here so that the inverse-step
    # arithmetic stays correct (#270 fix).
    domain = Domain()
    domain.add_int(
        "x", low=4, high=20, step=3
    )  # legal grid: 4, 7, 10, 13, 16, 19
    data = ExperimentData(domain=domain)
    sampler = create_sampler(sampler="random", seed=42)
    result = sampler.call(data=data, n_samples=20)

    df, _ = result.to_pandas()
    legal = {4, 7, 10, 13, 16, 19}
    for val in df["x"]:
        assert int(val) in legal


def test_sampler_categorical_only():
    domain = Domain()
    domain.add_category("cat", ["a", "b", "c"])

    data = ExperimentData(domain=domain)
    sampler = create_sampler(sampler="random", seed=42)
    result = sampler.call(data=data, n_samples=5)
    assert len(result) == 5
    df, _ = result.to_pandas()
    assert all(v in ["a", "b", "c"] for v in df["cat"])


# ======================= Latin array sampling =======================


class TestLatinArraySampling:
    def test_latin_sample_array_parameters_1d(self):
        input_space = {
            "arr": ArrayParameter(lower_bound=0.0, upper_bound=1.0, shape=(3,))
        }
        samples = latin_sample_array_parameters(
            input_space=input_space, n_samples=5, seed=42
        )
        assert len(samples) == 5
        for es in samples.values():
            arr = es.input_data["arr"]
            assert arr.shape == (3,)

    def test_latin_sample_array_parameters_2d(self):
        input_space = {
            "arr": ArrayParameter(
                lower_bound=0.0, upper_bound=1.0, shape=(2, 2)
            )
        }
        samples = latin_sample_array_parameters(
            input_space=input_space, n_samples=4, seed=42
        )
        assert len(samples) == 4
        for es in samples.values():
            arr = es.input_data["arr"]
            assert arr.shape == (2, 2)


# ======================= Sobol array sampling =======================


class TestSobolArraySampling:
    def test_sobol_sample_array_parameters_1d(self):
        input_space = {
            "arr": ArrayParameter(lower_bound=0.0, upper_bound=1.0, shape=(3,))
        }
        samples = sobol_sample_array_parameters(
            input_space=input_space, n_samples=4, seed=42
        )
        assert len(samples) == 4
        for es in samples.values():
            arr = es.input_data["arr"]
            assert arr.shape == (3,)

    def test_sobol_sample_array_parameters_2d(self):
        input_space = {
            "arr": ArrayParameter(
                lower_bound=0.0, upper_bound=1.0, shape=(2, 2)
            )
        }
        samples = sobol_sample_array_parameters(
            input_space=input_space, n_samples=4, seed=42
        )
        assert len(samples) == 4
        for es in samples.values():
            arr = es.input_data["arr"]
            assert arr.shape == (2, 2)


# ======================= Latin/Sobol with ArrayParameter domain =======================


def test_latin_sampler_with_array_domain():
    domain = Domain()
    domain.add_array("arr", low=0.0, high=1.0, shape=(2,))
    data = ExperimentData(domain=domain)
    sampler = create_sampler("latin", seed=42)
    result = sampler.call(data=data, n_samples=5)
    assert len(result) == 5


def test_sobol_sampler_with_array_domain():
    domain = Domain()
    domain.add_array("arr", low=0.0, high=1.0, shape=(2,))
    data = ExperimentData(domain=domain)
    sampler = create_sampler("sobol", seed=42)
    result = sampler.call(data=data, n_samples=4)
    assert len(result) == 4


# ======================= Grid value functions =======================


class TestGridValueFunctions:
    def test_grid_values_continuous_with_float_stepsize(self):
        input_space = {
            "x": ContinuousParameter(lower_bound=0.0, upper_bound=1.0)
        }
        result = grid_values_continuous_parameters(
            input_space=input_space,
            stepsize_continuous_parameters=0.25,
        )
        assert "x" in result
        assert len(result["x"]) == 4  # 0.0, 0.25, 0.5, 0.75

    def test_grid_values_continuous_with_dict_stepsize(self):
        input_space = {
            "x": ContinuousParameter(lower_bound=0.0, upper_bound=1.0)
        }
        result = grid_values_continuous_parameters(
            input_space=input_space,
            stepsize_continuous_parameters={"x": 0.5},
        )
        assert "x" in result
        assert len(result["x"]) == 2  # 0.0, 0.5

    def test_grid_values_continuous_dict_mismatched_raises(self):
        input_space = {
            "x": ContinuousParameter(lower_bound=0.0, upper_bound=1.0)
        }
        with pytest.raises(ValueError, match="stepsize_continuous_parameters"):
            grid_values_continuous_parameters(
                input_space=input_space,
                stepsize_continuous_parameters={"x": 0.5, "y": 0.5},
            )

    def test_grid_values_continuous_none_raises(self):
        # Issue #318: passing None with a non-empty input space used to
        # silently return an empty dict, which made the grid sampler emit
        # a degenerate single-row grid. It now raises a clear ValueError.
        input_space = {
            "x": ContinuousParameter(lower_bound=0.0, upper_bound=1.0)
        }
        with pytest.raises(ValueError, match="stepsize_continuous"):
            grid_values_continuous_parameters(
                input_space=input_space,
                stepsize_continuous_parameters=None,
            )

    def test_grid_values_continuous_none_empty_space_returns_empty(self):
        # When the input space has no continuous parameters at all, the
        # stepsize argument is genuinely irrelevant and the function
        # should return an empty dict without raising.
        result = grid_values_continuous_parameters(
            input_space={},
            stepsize_continuous_parameters=None,
        )
        assert result == {}

    def test_grid_values_discrete(self):
        input_space = {
            "d": DiscreteParameter(lower_bound=0, upper_bound=4, step=2)
        }
        result = grid_values_discrete_parameters(input_space=input_space)
        assert "d" in result
        assert list(result["d"]) == [0, 2, 4]

    def test_grid_values_categorical(self):
        input_space = {"cat": CategoricalParameter(categories=["a", "b", "c"])}
        result = grid_values_categorical_parameters(input_space=input_space)
        assert result == {"cat": ["a", "b", "c"]}

    def test_grid_values_constant(self):
        input_space = {"c": ConstantParameter(value=42)}
        result = grid_values_constant_parameters(input_space=input_space)
        assert result == {"c": [42]}


# ======================= Grid sampler integration =======================


def test_grid_sampler_mixed_domain():
    domain = Domain()
    domain.add_float("x", 0.0, 1.0)
    domain.add_int("d", 0, 2)
    domain.add_category("cat", ["a", "b"])

    data = ExperimentData(domain=domain)
    grid_sampler = Grid(stepsize_continuous_parameters=0.5)
    result = grid_sampler.call(data=data)
    # 2 continuous (0.0, 0.5) * 3 discrete (0,1,2) * 2 categorical (a,b) = 12
    assert len(result) == 12


def test_grid_sampler_continuous_only():
    domain = Domain()
    domain.add_float("x", 0.0, 1.0)
    domain.add_float("y", 0.0, 1.0)

    data = ExperimentData(domain=domain)
    grid_sampler = Grid(stepsize_continuous_parameters=0.5)
    result = grid_sampler.call(data=data)
    # 2 * 2 = 4
    assert len(result) == 4


def test_grid_sampler_constant_domain():
    domain = Domain()
    domain.add_constant("c", 10)

    data = ExperimentData(domain=domain)
    grid_sampler = Grid()
    result = grid_sampler.call(data=data)
    assert len(result) == 1
