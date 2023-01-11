import copy

import numpy as np
import pytest

from f3dasm.base.data import Data
from f3dasm.base.function import Function
from f3dasm.optimization.optimizer import Optimizer
from f3dasm.base.utils import make_nd_continuous_design
from f3dasm.functions import FUNCTIONS, FUNCTIONS_2D, Ackley, Levy, Sphere
from f3dasm.optimization import OPTIMIZERS
from f3dasm.optimization.cmaesadam import CMAESAdam
from f3dasm.sampling.randomuniform import RandomUniform


@pytest.mark.parametrize("function", FUNCTIONS_2D)
def test_plotting(function: Function):
    f = function(dimensionality=2)
    f.plot(px=10, show=False)


# @pytest.mark.smoke
# @pytest.mark.parametrize("seed", [42])
# @pytest.mark.parametrize("optimizer", [CMAESAdam])
# @pytest.mark.parametrize("function", [Levy])
# def test_all_optimizers_temp(seed: int, function: Function, optimizer: Optimizer):
#     test_all_optimizers_and_functions(seed, function, optimizer)


@pytest.mark.smoke
@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize("optimizer", OPTIMIZERS)
@pytest.mark.parametrize("function", [Levy, Ackley, Sphere])
def test_all_optimizers_3_functions(seed: int, function: Function, optimizer: Optimizer):
    test_all_optimizers_and_functions(seed, function, optimizer)


@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize("optimizer", OPTIMIZERS)
@pytest.mark.parametrize("function", FUNCTIONS)
def test_all_optimizers_and_functions(seed: int, function: Function, optimizer: Optimizer):
    i = 10  # iterations

    dim = 6
    if not function.is_dim_compatible(dim):
        dim = 4
        if not function.is_dim_compatible(dim):
            dim = 3
            if not function.is_dim_compatible(dim):
                dim = 2

    design = make_nd_continuous_design(bounds=np.tile([-1.0, 1.0], (dim, 1)), dimensionality=dim)

    # Sampler
    ran_sampler = RandomUniform(design=design, seed=seed)
    data = ran_sampler.get_samples(numsamples=30)

    func = function(noise=None, seed=seed, scale_bounds=np.tile([-1.0, 1.0], (dim, 1)), dimensionality=dim)

    # Evaluate the initial samples
    data.add_output(output=func(data), label="y")

    opt1 = optimizer(data=copy.copy(data), seed=seed)
    opt2 = optimizer(data=copy.copy(data), seed=seed)

    opt1.iterate(iterations=i, function=func)
    opt2.iterate(iterations=i, function=func)
    data1 = opt1.extract_data()
    data2 = opt2.extract_data()
    assert all(data1.data == data2.data)


# TODO: Use stored data to assess this property (maybe hypothesis ?)
@pytest.mark.smoke
@pytest.mark.parametrize("iterations", np.random.randint(low=1, high=100, size=5))
@pytest.mark.parametrize("optimizer", OPTIMIZERS)
@pytest.mark.parametrize("function", [Sphere])
def test_optimizer_iterations(iterations: int, function: Function, optimizer: Optimizer):
    numsamples = 40  # iterations
    seed = 42

    dim = 6
    if not function.is_dim_compatible(dim):
        dim = 4
        if not function.is_dim_compatible(dim):
            dim = 3
            if not function.is_dim_compatible(dim):
                dim = 2

    design = make_nd_continuous_design(bounds=np.tile([-1.0, 1.0], (dim, 1)), dimensionality=dim)

    # Sampler
    ran_sampler = RandomUniform(design=design, seed=seed)
    data: Data = ran_sampler.get_samples(numsamples=numsamples)

    func = function(noise=None, seed=seed, scale_bounds=np.tile([-1.0, 1.0], (dim, 1)), dimensionality=dim)

    # Evaluate the initial samples
    data.add_output(output=func(data), label="y")

    opt1: Optimizer = optimizer(data=data, seed=seed)

    opt1.iterate(iterations=iterations, function=func)

    data = opt1.extract_data()

    assert data.get_number_of_datapoints() == (iterations + numsamples)


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
