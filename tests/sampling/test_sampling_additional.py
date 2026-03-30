"""Additional sampling tests for edge cases and untested paths."""

import numpy as np
import pytest

from f3dasm import ExperimentData, create_sampler
from f3dasm._src.samplers import Sampler
from f3dasm.design import Domain

pytestmark = pytest.mark.smoke


@pytest.fixture
def mixed_domain():
    d = Domain()
    d.add_float(name="x", low=0.0, high=1.0)
    d.add_int(name="n", low=1, high=10)
    d.add_category(name="c", categories=["a", "b", "c"])
    d.add_output(name="y")
    return d


@pytest.fixture
def continuous_domain():
    d = Domain()
    d.add_float(name="x0", low=0.0, high=1.0)
    d.add_float(name="x1", low=0.0, high=1.0)
    d.add_output(name="y")
    return d


class TestCreateSampler:
    def test_create_sampler_returns_sampler(self):
        s = create_sampler("random", seed=42)
        assert isinstance(s, Sampler)

    def test_create_sampler_invalid_name(self):
        with pytest.raises(KeyError):
            create_sampler("invalid_sampler_name")

    def test_create_sampler_invalid_type(self):
        with pytest.raises(TypeError):
            create_sampler(12345)

    def test_create_sampler_rejects_non_str_non_config(self):
        s = create_sampler("random", seed=42)
        # Sampler object raises TypeError (only str/DictConfig)
        with pytest.raises(TypeError):
            create_sampler(s)

    @pytest.mark.parametrize("name", ["random", "latin", "sobol"])
    def test_all_builtin_samplers(self, name, continuous_domain):
        s = create_sampler(name, seed=42)
        data = ExperimentData(domain=continuous_domain)
        result = s.call(data, n_samples=8)
        assert len(result) == 8


class TestSamplerWithMixedDomain:
    def test_random_mixed_domain(self, mixed_domain):
        data = ExperimentData(domain=mixed_domain)
        sampler = create_sampler("random", seed=42)
        result = sampler.call(data=data, n_samples=10)
        assert len(result) == 10

    def test_latin_mixed_domain(self, mixed_domain):
        data = ExperimentData(domain=mixed_domain)
        sampler = create_sampler("latin", seed=42)
        result = sampler.call(data=data, n_samples=10)
        assert len(result) == 10


class TestSamplerReproducibility:
    def test_same_seed_same_result(self, continuous_domain):
        d1 = ExperimentData(domain=continuous_domain)
        s1 = create_sampler("random", seed=42)
        d1 = s1.call(data=d1, n_samples=5)

        d2 = ExperimentData(domain=continuous_domain)
        s2 = create_sampler("random", seed=42)
        d2 = s2.call(data=d2, n_samples=5)

        arr1, _ = d1.to_numpy()
        arr2, _ = d2.to_numpy()
        np.testing.assert_array_equal(arr1, arr2)

    def test_different_seed_different_result(self, continuous_domain):
        d1 = ExperimentData(domain=continuous_domain)
        s1 = create_sampler("random", seed=42)
        d1 = s1.call(data=d1, n_samples=5)

        d2 = ExperimentData(domain=continuous_domain)
        s2 = create_sampler("random", seed=99)
        d2 = s2.call(data=d2, n_samples=5)

        arr1, _ = d1.to_numpy()
        arr2, _ = d2.to_numpy()
        assert not np.array_equal(arr1, arr2)
