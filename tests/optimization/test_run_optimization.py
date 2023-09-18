import os
from random import Random

import numpy as np
import pytest

from f3dasm import run_multiple_realizations
from f3dasm.datageneration import (FUNCTIONS, FUNCTIONS_2D, FUNCTIONS_7D,
                                   Ackley, DataGenerator, Griewank, Levy,
                                   Rastrigin, Schwefel, Sphere)
from f3dasm.design import ExperimentData, make_nd_continuous_domain
from f3dasm.optimization import OPTIMIZERS, Optimizer
from f3dasm.sampling import RandomUniform


@pytest.mark.smoke
@pytest.mark.parametrize("optimizer", OPTIMIZERS)
@pytest.mark.parametrize("data_generator", [Levy, Ackley, Sphere])
@pytest.mark.parametrize("dimensionality", [2])
def test_run_multiple_realizations_3_functions(data_generator: DataGenerator,
                                               optimizer: Optimizer, dimensionality: int):
    test_run_multiple_realizations(data_generator, optimizer, dimensionality)


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
@pytest.mark.parametrize("data_generator", FUNCTIONS_2D)
@pytest.mark.parametrize("dimensionality", [2])
def test_run_multiple_realizations(data_generator: DataGenerator, optimizer: Optimizer, dimensionality: int):
    iterations = 30
    realizations = 3
    domain = np.tile([0.0, 1.0], (dimensionality, 1))

    domain = make_nd_continuous_domain(dimensionality=dimensionality, bounds=domain)
    func = data_generator(dimensionality=dimensionality, scale_bounds=domain.get_bounds())
    opt = optimizer(domain=domain)
    sampler = RandomUniform(domain=domain)

    # Check if os is windows
    if os.name == 'nt':
        PARALLELIZATION = False
    else:
        PARALLELIZATION = True

    if opt.get_name() in ['EvoSaxCMAES', 'EvoSaxSimAnneal', 'EvoSaxPSO', 'EvoSaxDE']:
        PARALLELIZATION = False

    res = run_multiple_realizations(
        optimizer=opt,
        data_generator=func,
        sampler=sampler,
        iterations=iterations,
        realizations=realizations,
        parallelization=PARALLELIZATION,
    )


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
@pytest.mark.parametrize("data_generator", FUNCTIONS_7D)
@pytest.mark.parametrize("dimensionality", [7])
def test_run_multiple_realizations_7D(data_generator: DataGenerator, optimizer: Optimizer, dimensionality: int):
    test_run_multiple_realizations(data_generator, optimizer, dimensionality)


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
@pytest.mark.parametrize("data_generator", [Griewank])
@pytest.mark.parametrize("dimensionality", [2])
def test_run_multiple_realizations_fast(data_generator: DataGenerator, optimizer: Optimizer, dimensionality: int):
    test_run_multiple_realizations(data_generator, optimizer, dimensionality)


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
