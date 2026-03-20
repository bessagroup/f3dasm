"""Tests for benchmark functions in f3dasm."""

import numpy as np
import pytest

from f3dasm._src.datageneration.benchmarkfunctions import (
    BENCHMARK_BOUNDS,
    BENCHMARK_FUNCTIONS,
    ackley,
    rastrigin,
    rosenbrock,
    sphere,
)

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize("name", list(BENCHMARK_FUNCTIONS.keys()))
def test_all_benchmark_functions_return_scalar(name):
    """Every benchmark function should return a scalar float."""
    func = BENCHMARK_FUNCTIONS[name]
    bounds = BENCHMARK_BOUNDS[name]
    # Use midpoint of bounds as test input (2D)
    mid = (bounds[0] + bounds[1]) / 2
    x = np.array([mid, mid])
    result = func(x)
    assert np.isscalar(result) or isinstance(result, np.floating)


def test_benchmark_bounds_match_functions():
    """Every function in BENCHMARK_FUNCTIONS should have bounds."""
    assert set(BENCHMARK_FUNCTIONS.keys()) == set(BENCHMARK_BOUNDS.keys())


def test_sphere_at_origin():
    """sphere(0, 0) = 0"""
    assert sphere(np.array([0.0, 0.0])) == 0.0


def test_sphere_known_value():
    """sphere(1, 1) = 2"""
    assert sphere(np.array([1.0, 1.0])) == 2.0


def test_ackley_at_origin():
    """ackley(0, 0) ≈ 0"""
    result = ackley(np.array([0.0, 0.0]))
    assert abs(result) < 1e-10


def test_rosenbrock_at_optimum():
    """rosenbrock(1, 1) = 0"""
    result = rosenbrock(np.array([1.0, 1.0]))
    assert abs(result) < 1e-10


def test_rastrigin_at_origin():
    """rastrigin(0, 0) = 0"""
    result = rastrigin(np.array([0.0, 0.0]))
    assert abs(result) < 1e-10


def test_sphere_higher_dimensions():
    """sphere works with arbitrary dimensions."""
    x = np.zeros(10)
    assert sphere(x) == 0.0
    x = np.ones(10)
    assert sphere(x) == 10.0


@pytest.mark.parametrize("name", list(BENCHMARK_FUNCTIONS.keys()))
def test_benchmark_functions_positive_at_random_point(name):
    """Most benchmark functions should not crash with random input."""
    func = BENCHMARK_FUNCTIONS[name]
    bounds = BENCHMARK_BOUNDS[name]
    rng = np.random.default_rng(42)
    x = rng.uniform(bounds[0], bounds[1], size=2)
    result = func(x)
    assert np.isfinite(result)
