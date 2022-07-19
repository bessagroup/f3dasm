from typing import Tuple
import pytest

from f3dasm.base.designofexperiments import make_nd_continuous_design
from f3dasm.base.optimization import Optimizer
from f3dasm.base.simulation import Function
from f3dasm.optimization.pygmo_implementations import CMAES
from f3dasm.sampling.samplers import RandomUniformSampling
from f3dasm.simulation.benchmark_functions import Levy


@pytest.fixture
def optimizer_and_function_42():
    seed = 42
    design = make_nd_continuous_design(bounds=[-1.0, 1.0], dimensions=10)

    # Sampler
    ran_sampler = RandomUniformSampling(doe=design, seed=seed)
    data = ran_sampler.get_samples(numsamples=30)

    levy = Levy(noise=False, seed=42)

    # Evaluate the initial samples
    data.add_output(output=levy.eval(data), label="y")

    # algorithm
    cmaes = CMAES(data=data, seed=seed)
    return cmaes, levy


@pytest.fixture
def optimizer_and_function_999():
    seed = 999
    design = make_nd_continuous_design(bounds=[-1.0, 1.0], dimensions=10)

    # Sampler
    ran_sampler = RandomUniformSampling(doe=design, seed=seed)
    data = ran_sampler.get_samples(numsamples=30)

    levy = Levy(noise=False, seed=42)

    # Evaluate the initial samples
    data.add_output(output=levy.eval(data), label="y")

    # algorithm
    cmaes = CMAES(data=data, seed=seed)
    return cmaes, levy


def test_cmaes_same_seeding(optimizer_and_function_42: Tuple[Optimizer, Function]):
    cmaes, levy = optimizer_and_function_42
    cmaes2, levy2 = optimizer_and_function_42
    i = 100
    cmaes.iterate(iterations=i, function=levy)
    cmaes2.iterate(iterations=i, function=levy2)
    data = cmaes.extract_data()
    data2 = cmaes2.extract_data()

    assert all(data.data == data2.data)


def test_cmaes_different_seeding(
    optimizer_and_function_42: Tuple[Optimizer, Function],
    optimizer_and_function_999: Tuple[Optimizer, Function],
):
    cmaes, levy = optimizer_and_function_42
    cmaes2, levy2 = optimizer_and_function_999
    i = 100
    cmaes.iterate(iterations=i, function=levy)
    cmaes2.iterate(iterations=i, function=levy2)
    data = cmaes.extract_data()
    data2 = cmaes2.extract_data()

    assert any(data.data != data2.data)
