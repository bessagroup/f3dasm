import copy
from typing import List

import numpy as np
import pytest

from f3dasm.datageneration import DataGenerator
from f3dasm.datageneration.functions import (FUNCTIONS, FUNCTIONS_2D, Ackley,
                                             Levy, Sphere)
from f3dasm.design import ExperimentData, make_nd_continuous_domain
from f3dasm.optimization import OPTIMIZERS, Optimizer


@pytest.mark.smoke
@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_get_info(data: ExperimentData, optimizer: Optimizer):
    opt: Optimizer = optimizer(data.domain)
    characteristics = opt.get_info()
    assert isinstance(characteristics, List)


@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize("optimizer", OPTIMIZERS)
@pytest.mark.parametrize("data_generator", FUNCTIONS)
def test_all_optimizers_and_functions(seed: int, data_generator: DataGenerator, optimizer: Optimizer):
    i = 10  # iterations

    dim = 6
    if not data_generator.is_dim_compatible(dim):
        dim = 4
        if not data_generator.is_dim_compatible(dim):
            dim = 3
            if not data_generator.is_dim_compatible(dim):
                dim = 2

    domain = make_nd_continuous_domain(bounds=np.tile([-1.0, 1.0], (dim, 1)), dimensionality=dim)

    # Sampler

    data1 = ExperimentData.from_sampling(sampler='random', domain=domain, n_samples=30, seed=seed)
    data2 = ExperimentData.from_sampling(sampler='random', domain=domain, n_samples=30, seed=seed)

    func = data_generator(noise=None, seed=seed, scale_bounds=np.tile([-1.0, 1.0], (dim, 1)), dimensionality=dim)

    data1.evaluate(func)
    data2.evaluate(func)

    opt1 = optimizer(domain=domain, seed=seed)

    data1.optimize(optimizer=opt1, data_generator=func, iterations=i)
    data2.optimize(optimizer=opt1, data_generator=func, iterations=i)

    assert (data1 == data2)


@pytest.mark.smoke
@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize("optimizer", OPTIMIZERS)
@pytest.mark.parametrize("data_generator", [Levy, Ackley, Sphere])
def test_all_optimizers_3_functions(seed: int, data_generator: DataGenerator, optimizer: Optimizer):
    test_all_optimizers_and_functions(seed, data_generator, optimizer)


# TODO: Use stored data to assess this property (maybe hypothesis ?)
@pytest.mark.smoke
@pytest.mark.parametrize("iterations", [10, 23, 66, 86])
@pytest.mark.parametrize("optimizer", OPTIMIZERS)
@pytest.mark.parametrize("data_generator", [Sphere])
def test_optimizer_iterations(iterations: int, data_generator: DataGenerator, optimizer: Optimizer):
    numsamples = 40  # initial samples
    seed = 42

    dim = 6
    if not data_generator.is_dim_compatible(dim):
        dim = 4
        if not data_generator.is_dim_compatible(dim):
            dim = 3
            if not data_generator.is_dim_compatible(dim):
                dim = 2

    domain = make_nd_continuous_domain(bounds=np.tile([-1.0, 1.0], (dim, 1)), dimensionality=dim)

    data = ExperimentData.from_sampling(sampler='random', domain=domain, n_samples=numsamples, seed=seed)

    func = data_generator(noise=None, seed=seed, scale_bounds=np.tile([-1.0, 1.0], (dim, 1)), dimensionality=dim)

    # Evaluate the initial samples
    data.evaluate(func, mode='sequential')

    opt1: Optimizer = optimizer(domain=domain, seed=seed)

    data.optimize(optimizer=opt1, data_generator=func, iterations=iterations)

    assert len(data) == (iterations + numsamples)


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
