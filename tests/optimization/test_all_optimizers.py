from __future__ import annotations

from typing import List

import numpy as np
import pytest

from f3dasm import ExperimentData
from f3dasm._src.datageneration.functions.function_factory import \
    is_dim_compatible
from f3dasm._src.optimization.optimizer_factory import optimizer_factory
from f3dasm.datageneration import DataGenerator
from f3dasm.datageneration.functions import FUNCTIONS
from f3dasm.design import make_nd_continuous_domain
from f3dasm.optimization import OPTIMIZERS, Optimizer


@pytest.mark.smoke
@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_get_info(data: ExperimentData, optimizer: str):
    opt: Optimizer = optimizer_factory(optimizer, data.domain)
    characteristics = opt.get_info()
    assert isinstance(characteristics, List)


@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize("optimizer", OPTIMIZERS)
@pytest.mark.parametrize("data_generator", FUNCTIONS)
def test_all_optimizers_and_functions(seed: int, data_generator: str, optimizer: str):
    i = 10  # iterations

    dim = 6
    domain = make_nd_continuous_domain(bounds=np.tile([-1.0, 1.0], (dim, 1)), dimensionality=dim)
    if not is_dim_compatible(data_generator, domain):
        dim = 4
        domain = make_nd_continuous_domain(bounds=np.tile([-1.0, 1.0], (dim, 1)), dimensionality=dim)
        if not is_dim_compatible(data_generator, domain):
            dim = 3
            domain = make_nd_continuous_domain(bounds=np.tile([-1.0, 1.0], (dim, 1)), dimensionality=dim)
            if not is_dim_compatible(data_generator, domain):
                dim = 2
                domain = make_nd_continuous_domain(bounds=np.tile([-1.0, 1.0], (dim, 1)), dimensionality=dim)

    # Sampler

    data1 = ExperimentData.from_sampling(sampler='random', domain=domain, n_samples=30, seed=seed)
    data2 = ExperimentData.from_sampling(sampler='random', domain=domain, n_samples=30, seed=seed)

    data1.evaluate(data_generator, kwargs={'noise': None, 'seed': seed,
                   'scale_bounds': np.tile([-1.0, 1.0], (dim, 1))})
    data2.evaluate(data_generator, kwargs={'noise': None, 'seed': seed,
                   'scale_bounds': np.tile([-1.0, 1.0], (dim, 1))})

    data1.optimize(optimizer=optimizer, data_generator=data_generator,
                   iterations=i, kwargs={'noise': None, 'seed': seed,
                                         'scale_bounds': np.tile([-1.0, 1.0], (dim, 1))},
                   hyperparameters={'seed': seed})
    data2.optimize(optimizer=optimizer, data_generator=data_generator,
                   iterations=i, kwargs={'noise': None, 'seed': seed,
                                         'scale_bounds': np.tile([-1.0, 1.0], (dim, 1))},
                   hyperparameters={'seed': seed})

    assert (data1 == data2)


@pytest.mark.smoke
@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize("optimizer", OPTIMIZERS)
@pytest.mark.parametrize("data_generator", ['levy', 'ackley', 'sphere'])
def test_all_optimizers_3_functions(seed: int, data_generator: DataGenerator, optimizer: str):
    test_all_optimizers_and_functions(seed, data_generator, optimizer)


# TODO: Use stored data to assess this property (maybe hypothesis ?)
@pytest.mark.smoke
@pytest.mark.parametrize("iterations", [10, 23, 66, 86])
@pytest.mark.parametrize("optimizer", OPTIMIZERS)
@pytest.mark.parametrize("data_generator", ["sphere"])
def test_optimizer_iterations(iterations: int, data_generator: str, optimizer: str):
    numsamples = 40  # initial samples
    seed = 42

    dim = 6
    domain = make_nd_continuous_domain(bounds=np.tile([-1.0, 1.0], (dim, 1)), dimensionality=dim)
    if not is_dim_compatible(data_generator, domain):
        dim = 4
        domain = make_nd_continuous_domain(bounds=np.tile([-1.0, 1.0], (dim, 1)), dimensionality=dim)
        if not is_dim_compatible(data_generator, domain):
            dim = 3
            domain = make_nd_continuous_domain(bounds=np.tile([-1.0, 1.0], (dim, 1)), dimensionality=dim)
            if not is_dim_compatible(data_generator, domain):
                dim = 2
                domain = make_nd_continuous_domain(bounds=np.tile([-1.0, 1.0], (dim, 1)), dimensionality=dim)

    data = ExperimentData.from_sampling(sampler='random', domain=domain, n_samples=numsamples, seed=seed)

    # func = data_generator(noise=None, seed=seed, scale_bounds=np.tile([-1.0, 1.0], (dim, 1)), dimensionality=dim)

    # Evaluate the initial samples
    data.evaluate(data_generator, mode='sequential', kwargs={'seed': seed, 'noise': None,
                                                             'scale_bounds': np.tile([-1.0, 1.0], (dim, 1)), })

    data.optimize(optimizer=optimizer, data_generator=data_generator,
                  iterations=iterations, kwargs={'seed': seed, 'noise': None,
                                                 'scale_bounds': np.tile([-1.0, 1.0], (dim, 1)), },
                  hyperparameters={'seed': seed})

    assert len(data) == (iterations + numsamples)


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
