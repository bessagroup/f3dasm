from random import Random
import numpy as np
import pytest
from f3dasm.base.data import Data
from f3dasm.base.function import Function
from f3dasm.base.optimization import Optimizer
from f3dasm.base.utils import make_nd_continuous_design

from f3dasm.run_optimization import run_multiple_realizations
from f3dasm.functions import FUNCTIONS_2D, FUNCTIONS, FUNCTIONS_7D, Ackley, Levy, Griewank, Schwefel, Rastrigin, Sphere
from f3dasm.optimization import OPTIMIZERS
from f3dasm.sampling.randomuniform import RandomUniform


@pytest.mark.smoke
@pytest.mark.parametrize("optimizer", OPTIMIZERS)
@pytest.mark.parametrize("function", [Levy, Ackley, Sphere])
@pytest.mark.parametrize("dimensionality", [2])
def test_run_multiple_realizations_3_functions(function: Function, optimizer: Optimizer, dimensionality: int):
    test_run_multiple_realizations(function, optimizer, dimensionality)


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
@pytest.mark.parametrize("function", FUNCTIONS_2D)
@pytest.mark.parametrize("dimensionality", [2])
def test_run_multiple_realizations(function: Function, optimizer: Optimizer, dimensionality: int):
    iterations = 30
    realizations = 3
    domain = np.tile([0.0, 1.0], (dimensionality, 1))

    design = make_nd_continuous_design(dimensionality=dimensionality, bounds=domain)
    func = function(dimensionality=dimensionality, scale_bounds=domain)
    data = Data(design=design)
    opt = optimizer(data=data)
    sampler = RandomUniform(design=design)

    res = run_multiple_realizations(
        optimizer=opt,
        function=func,
        sampler=sampler,
        iterations=iterations,
        realizations=realizations,
    )


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
@pytest.mark.parametrize("function", FUNCTIONS_7D)
@pytest.mark.parametrize("dimensionality", [7])
def test_run_multiple_realizations_7D(function: Function, optimizer: Optimizer, dimensionality: int):
    test_run_multiple_realizations(function, optimizer, dimensionality)


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
@pytest.mark.parametrize("function", [Griewank])
@pytest.mark.parametrize("dimensionality", [2])
def test_run_multiple_realizations_fast(function: Function, optimizer: Optimizer, dimensionality: int):
    test_run_multiple_realizations(function, optimizer, dimensionality)


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
