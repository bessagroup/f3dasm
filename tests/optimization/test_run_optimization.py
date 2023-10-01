import os

import numpy as np
import pytest

from f3dasm import run_multiple_realizations
from f3dasm.datageneration import DataGenerator
from f3dasm.datageneration.functions import FUNCTIONS_2D, FUNCTIONS_7D
from f3dasm.design import make_nd_continuous_domain
from f3dasm.optimization import OPTIMIZERS, Optimizer


@pytest.mark.smoke
@pytest.mark.parametrize("optimizer", OPTIMIZERS)
@pytest.mark.parametrize("data_generator", ['Levy', 'Ackley', 'Sphere'])
@pytest.mark.parametrize("dimensionality", [2])
def test_run_multiple_realizations_3_functions(data_generator: DataGenerator,
                                               optimizer: Optimizer, dimensionality: int):
    test_run_multiple_realizations(data_generator, optimizer, dimensionality)


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
@pytest.mark.parametrize("data_generator", FUNCTIONS_2D)
@pytest.mark.parametrize("dimensionality", [2])
def test_run_multiple_realizations(data_generator: str, optimizer: Optimizer, dimensionality: int):
    iterations = 30
    realizations = 3
    domain = np.tile([0.0, 1.0], (dimensionality, 1))

    domain = make_nd_continuous_domain(dimensionality=dimensionality, bounds=domain)

    kwargs = {'scale_bounds': domain.get_bounds()}
    opt = optimizer(domain=domain)
    sampler = 'random'

    # Check if os is windows
    if os.name == 'nt':
        PARALLELIZATION = False
    else:
        PARALLELIZATION = True

    if opt.get_name() in ['EvoSaxCMAES', 'EvoSaxSimAnneal', 'EvoSaxPSO', 'EvoSaxDE']:
        PARALLELIZATION = False

    _ = run_multiple_realizations(
        optimizer=opt,
        data_generator=data_generator,
        kwargs=kwargs,
        sampler=sampler,
        domain=domain,
        iterations=iterations,
        realizations=realizations,
        parallelization=PARALLELIZATION,
    )


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
@pytest.mark.parametrize("data_generator", FUNCTIONS_7D)
@pytest.mark.parametrize("dimensionality", [7])
def test_run_multiple_realizations_7D(data_generator: str, optimizer: Optimizer, dimensionality: int):
    test_run_multiple_realizations(data_generator, optimizer, dimensionality)


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
@pytest.mark.parametrize("data_generator", ['griewank'])
@pytest.mark.parametrize("dimensionality", [2])
def test_run_multiple_realizations_fast(data_generator: str, optimizer: Optimizer, dimensionality: int):
    test_run_multiple_realizations(data_generator, optimizer, dimensionality)


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
