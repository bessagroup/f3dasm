from random import Random
import numpy as np
import pytest
from f3dasm.base.data import Data
from f3dasm.base.function import Function
from f3dasm.base.optimization import Optimizer
from f3dasm.base.utils import make_nd_continuous_design

from f3dasm.run_optimization import run_multiple_realizations
from f3dasm.functions import FUNCTIONS_2D, FUNCTIONS, Ackley, Levy, Griewank, Schwefel, Rastrigin, Sphere
from f3dasm.optimization import OPTIMIZERS
from f3dasm.sampling.samplers import RandomUniformSampling


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
@pytest.mark.parametrize("function", FUNCTIONS_2D)
def test_run_multiple_realizations(function: Function, optimizer: Optimizer):
    iterations = 30
    realizations = 3

    design = make_nd_continuous_design(dimensions=2, bounds=np.tile([0.0, 1.0], (2, 1)))
    func = function(noise=False, dimensionality=2)
    data = Data(designspace=design)
    opt = optimizer(data=data)
    sampler = RandomUniformSampling(doe=design)

    res = run_multiple_realizations(
        optimizer=opt,
        function=func,
        sampler=sampler,
        iterations=iterations,
        realizations=realizations,
    )


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
@pytest.mark.parametrize("function", [Griewank])
def test_run_multiple_realizations_fast(function: Function, optimizer: Optimizer):
    test_run_multiple_realizations(function, optimizer)


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
