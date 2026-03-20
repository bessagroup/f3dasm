"""Extended tests for sampling methods."""

import numpy as np
import pytest

from f3dasm import ExperimentData, create_sampler
from f3dasm._src.samplers import (
    _stretch_samples,
    next_power_of_two,
    sample_constant,
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


# ======================= sample_constant =======================


def test_sample_constant():
    domain = Domain()
    domain.add_constant("c0", 42)
    domain.add_constant("c1", "hello")

    samples = sample_constant(domain, n_samples=5)
    assert samples.shape == (5, 2)
    # Values are stored as objects, compare as strings
    assert all(str(samples[i, 0]) == "42" for i in range(5))
    assert all(samples[i, 1] == "hello" for i in range(5))


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


def test_sampler_categorical_only():
    domain = Domain()
    domain.add_category("cat", ["a", "b", "c"])

    data = ExperimentData(domain=domain)
    sampler = create_sampler(sampler="random", seed=42)
    result = sampler.call(data=data, n_samples=5)
    assert len(result) == 5
    df, _ = result.to_pandas()
    assert all(v in ["a", "b", "c"] for v in df["cat"])
