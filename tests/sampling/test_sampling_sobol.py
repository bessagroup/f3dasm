"""Tests for the Sobol sampler (issue #358)."""

import numpy as np
import pytest

from f3dasm import ExperimentData, create_sampler
from f3dasm.design import Domain

pytestmark = pytest.mark.smoke


def _sobol(domain: Domain, seed, n_samples: int):
    """Draw ``n_samples`` Sobol points and return the input DataFrame."""
    samples = create_sampler("sobol_sampler", seed=seed).call(
        data=ExperimentData(domain=domain), n_samples=n_samples
    )
    df, _ = samples.to_pandas()
    return df


def _float_domain(dim: int, low: float = -1.0, high: float = 1.0) -> Domain:
    domain = Domain()
    for k in range(dim):
        domain.add_float(f"x{k}", low, high)
    return domain


def _columns(df, names) -> np.ndarray:
    return df[list(names)].to_numpy(dtype=float)


def _array_values(df, name) -> np.ndarray:
    return np.stack(df[name].to_list()).astype(float)


def _is_stratified(values: np.ndarray, low, high) -> bool:
    """One point in each of ``n`` equal bins, per coordinate.

    Every one-dimensional projection of ``n = 2**m`` scrambled Sobol
    points has this property; repeated or clustered coordinates do not.
    """
    n = values.shape[0]
    bins = np.floor((values - low) / (high - low) * n).astype(int)
    return all(
        sorted(bins[:, j]) == list(range(n)) for j in range(values.shape[1])
    )


def test_seed_changes_the_design():
    domain = _float_domain(5)
    names = list(domain.input_names)
    a = _columns(_sobol(domain, seed=0, n_samples=8), names)
    b = _columns(_sobol(domain, seed=123, n_samples=8), names)
    assert not np.allclose(a, b)


def test_same_seed_reproduces_the_design():
    domain = _float_domain(5)
    names = list(domain.input_names)
    a = _columns(_sobol(domain, seed=7, n_samples=8), names)
    b = _columns(_sobol(domain, seed=7, n_samples=8), names)
    np.testing.assert_array_equal(a, b)


def test_no_seed_gives_fresh_designs():
    domain = _float_domain(3)
    names = list(domain.input_names)
    a = _columns(_sobol(domain, seed=None, n_samples=8), names)
    b = _columns(_sobol(domain, seed=None, n_samples=8), names)
    assert not np.allclose(a, b)


def test_design_is_scrambled():
    """The unscrambled sequence starts at the lower corner, then the centre,
    and its early rows have equal columns."""
    domain = _float_domain(5)
    values = _columns(_sobol(domain, seed=0, n_samples=4), domain.input_names)
    assert not np.allclose(values[0], -1.0)
    assert not np.allclose(values[1], 0.0)
    assert all(len(np.unique(row)) == values.shape[1] for row in values)


def test_continuous_points_are_stratified_and_inside_bounds():
    domain = _float_domain(4, low=2.0, high=5.0)
    values = _columns(_sobol(domain, seed=1, n_samples=16), domain.input_names)
    assert values.min() >= 2.0 and values.max() <= 5.0
    assert _is_stratified(values, 2.0, 5.0)


def test_array_points_are_stratified():
    """Array parameters get Sobol points, not a sensitivity-analysis design
    that repeats coordinates across rows."""
    domain = Domain()
    domain.add_array("a", shape=(2, 2), low=0.0, high=1.0)
    df = _sobol(domain, seed=3, n_samples=8)
    values = _array_values(df, "a")
    assert values.shape == (8, 2, 2)
    assert _is_stratified(values.reshape(8, -1), 0.0, 1.0)


def test_array_respects_per_element_bounds():
    low = np.array([0.0, 10.0, -5.0])
    high = np.array([1.0, 20.0, 5.0])
    domain = Domain()
    domain.add_array("a", shape=(3,), low=low, high=high)
    values = _array_values(_sobol(domain, seed=0, n_samples=8), "a")
    assert np.all(values >= low) and np.all(values <= high)
    assert _is_stratified(values, low, high)


def test_parameters_of_equal_size_are_not_copies():
    """Continuous and array parameters share one seed; they must still get
    different coordinates."""
    domain = _float_domain(2, low=0.0, high=1.0)
    domain.add_array("a", shape=(2,), low=0.0, high=1.0)
    domain.add_array("b", shape=(2,), low=0.0, high=1.0)
    df = _sobol(domain, seed=0, n_samples=8)
    x = _columns(df, ["x0", "x1"])
    a = _array_values(df, "a")
    b = _array_values(df, "b")
    assert not np.allclose(x, a)
    assert not np.allclose(a, b)
    assert _is_stratified(np.hstack([x, a, b]), 0.0, 1.0)


def test_mixed_domain_samples_every_parameter():
    domain = Domain()
    domain.add_float("x", 0.0, 1.0)
    domain.add_int("k", 0, 3)
    domain.add_array("a", shape=(2,), low=0.0, high=1.0)
    domain.add_category("c", ["p", "q"])
    df = _sobol(domain, seed=0, n_samples=4)
    assert set(df.columns) == {"x", "k", "a", "c"}
    assert _array_values(df, "a").shape == (4, 2)
    assert df["k"].isin(range(4)).all()
    assert df["c"].isin(["p", "q"]).all()


def test_sobol_without_scipy_raises_clear_error(monkeypatch):
    """scipy is an optional extra (issue #301); the Sobol sampler must say
    how to install it rather than fail with a bare NameError."""
    import f3dasm._src.samplers as samplers_mod

    monkeypatch.setattr(samplers_mod, "qmc", None)
    with pytest.raises(ImportError, match=r"f3dasm\[scipy\]"):
        _sobol(_float_domain(2), seed=0, n_samples=4)


def test_sobol_with_old_scipy_raises_clear_error(monkeypatch):
    """scipy < 1.15 has no ``rng`` keyword on ``qmc.Sobol``."""
    import f3dasm._src.samplers as samplers_mod

    class _OldSobol:
        def __init__(self, d, *, scramble=True, seed=None):
            pass

    class _OldQmc:
        Sobol = _OldSobol

    monkeypatch.setattr(samplers_mod, "qmc", _OldQmc)
    with pytest.raises(ImportError, match=r"scipy>=1\.15"):
        _sobol(_float_domain(2), seed=0, n_samples=4)
