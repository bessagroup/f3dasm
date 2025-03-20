from __future__ import annotations

import numpy as np
import pytest

from f3dasm import ExperimentData, create_sampler
from f3dasm._src.datageneration.datagenerator_factory import \
    create_datagenerator
from f3dasm._src.optimization.optimizer_factory import create_optimizer
from f3dasm.datageneration import DataGenerator
from f3dasm.datageneration.functions import FUNCTIONS
from f3dasm.design import make_nd_continuous_domain
from f3dasm.optimization import available_optimizers


@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize("optimizer", available_optimizers())
@pytest.mark.parametrize("data_generator", FUNCTIONS)
def test_all_optimizers_and_functions(seed: int, data_generator: str, optimizer: str):
    i = 10  # iterations

    _func = create_datagenerator(
        data_generator=data_generator, seed=seed)

    dim = 6
    domain = make_nd_continuous_domain(bounds=np.tile(
        [-1.0, 1.0], (dim, 1)), dimensionality=dim)
    if not _func.is_dim_compatible(d=dim):
        dim = 4
        domain = make_nd_continuous_domain(bounds=np.tile(
            [-1.0, 1.0], (dim, 1)), dimensionality=dim)
        if not _func.is_dim_compatible(d=dim):
            dim = 3
            domain = make_nd_continuous_domain(bounds=np.tile(
                [-1.0, 1.0], (dim, 1)), dimensionality=dim)
            if not _func.is_dim_compatible(d=dim):
                dim = 2
                domain = make_nd_continuous_domain(bounds=np.tile(
                    [-1.0, 1.0], (dim, 1)), dimensionality=dim)

    _func = create_datagenerator(
        data_generator=data_generator, seed=seed,
        scale_bounds=np.tile([-1.0, 1.0], (dim, 1)),
    )

    sampler = create_sampler(sampler='random', seed=seed)
    data1 = ExperimentData(domain=domain)
    data1 = sampler.call(data=data1, n_samples=30)

    data2 = ExperimentData(domain=domain)
    data2 = sampler.call(data=data2, n_samples=30)

    _func.arm(data=data1)
    data1 = _func.call(data=data1, mode='sequential')

    data2 = _func.call(data=data2, mode='sequential')

    _optimizer1 = create_optimizer(optimizer=optimizer, seed=seed)
    _optimizer2 = create_optimizer(optimizer=optimizer, seed=seed)

    data1.optimize(optimizer=_optimizer1, data_generator=_func,
                   iterations=i)
    data2.optimize(optimizer=_optimizer2, data_generator=_func,
                   iterations=i)

    data1.replace_nan(None)
    data2.replace_nan(None)

    assert (data1 == data2)


@pytest.mark.smoke
@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize("optimizer", available_optimizers())
@pytest.mark.parametrize("data_generator", ['levy', 'ackley', 'sphere'])
def test_all_optimizers_3_functions(seed: int, data_generator: DataGenerator, optimizer: str):
    test_all_optimizers_and_functions(seed, data_generator, optimizer)


# TODO: Use stored data to assess this property (maybe hypothesis ?)
@pytest.mark.smoke
@pytest.mark.parametrize("iterations", [10, 23, 66, 86])
@pytest.mark.parametrize("optimizer", available_optimizers())
@pytest.mark.parametrize("data_generator", ["sphere"])
@pytest.mark.parametrize("x0_selection", ["best", "new"])
def test_optimizer_iterations(iterations: int, data_generator: str,
                              optimizer: str, x0_selection: str):
    numsamples = 40  # initial samples
    seed = 42

    _func = create_datagenerator(
        data_generator=data_generator,
        seed=seed, noise=None,
    )

    dim = 6
    domain = make_nd_continuous_domain(bounds=np.tile(
        [-1.0, 1.0], (dim, 1)), dimensionality=dim)
    if not _func.is_dim_compatible(d=dim):
        dim = 4
        domain = make_nd_continuous_domain(bounds=np.tile(
            [-1.0, 1.0], (dim, 1)), dimensionality=dim)
        if not _func.is_dim_compatible(d=dim):
            dim = 3
            domain = make_nd_continuous_domain(bounds=np.tile(
                [-1.0, 1.0], (dim, 1)), dimensionality=dim)
            if not _func.is_dim_compatible(d=dim):
                dim = 2
                domain = make_nd_continuous_domain(bounds=np.tile(
                    [-1.0, 1.0], (dim, 1)), dimensionality=dim)

    sampler = create_sampler(sampler='random', seed=seed)
    data = ExperimentData(domain=domain)

    data = sampler.call(data=data, n_samples=numsamples)
    _func = create_datagenerator(
        data_generator=data_generator,
        seed=seed, noise=None,
        scale_bounds=np.tile([-1.0, 1.0], (dim, 1)),
    )

    _func.arm(data=data)

    # Evaluate the initial samples
    data = _func.call(data=data, mode='sequential')

    _optimizer = create_optimizer(optimizer, seed=seed)
    population = _optimizer._population if hasattr(
        _optimizer, '_population') else 1

    if x0_selection == "new":
        sampler = create_sampler(sampler='random', seed=seed)
    else:
        sampler = None

    if x0_selection == "new" and iterations < population:
        with pytest.raises(ValueError):
            data.optimize(
                optimizer=_optimizer, data_generator=_func,
                iterations=iterations,
                x0_selection=x0_selection,
                sampler=sampler,
            )

    else:
        data.optimize(
            optimizer=_optimizer, data_generator=_func,
            iterations=iterations,
            x0_selection=x0_selection,
            sampler=sampler,
        )

        assert len(data) == (iterations + numsamples)


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
