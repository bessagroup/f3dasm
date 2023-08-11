import copy
from typing import List

import numpy as np
import pytest

from f3dasm.datageneration.functions import (FUNCTIONS, FUNCTIONS_2D, Ackley,
                                             Levy, Sphere)
from f3dasm.datageneration.functions.function import Function
from f3dasm.design import make_nd_continuous_domain
from f3dasm.design.experimentdata import ExperimentData
from f3dasm.optimization import OPTIMIZERS
from f3dasm.optimization.optimizer import Optimizer
from f3dasm.sampling.randomuniform import RandomUniform


@pytest.mark.smoke
@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_get_info(data: ExperimentData, optimizer: Optimizer):
    opt: Optimizer = optimizer(data=data)
    characteristics = opt.get_info()
    assert isinstance(characteristics, List)


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

    design = make_nd_continuous_domain(bounds=np.tile([-1.0, 1.0], (dim, 1)), dimensionality=dim)

    # Sampler
    ran_sampler = RandomUniform(design=design, seed=seed)
    data = ran_sampler.get_samples(numsamples=30)

    func = function(noise=None, seed=seed, scale_bounds=np.tile([-1.0, 1.0], (dim, 1)), dimensionality=dim)

    # Evaluate the initial samples
    data.fill_output(output=func(data), label="y")

    opt1 = optimizer(data=copy.copy(data), seed=seed)
    opt2 = optimizer(data=copy.copy(data), seed=seed)

    opt1.iterate(iterations=i, function=func)
    opt2.iterate(iterations=i, function=func)
    data1 = opt1.extract_data()
    data2 = opt2.extract_data()
    assert all(data1.input_data.data == data2.input_data.data)
    assert all(data1.output_data.data == data2.output_data.data)


@pytest.mark.smoke
@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize("optimizer", OPTIMIZERS)
@pytest.mark.parametrize("function", [Levy, Ackley, Sphere])
def test_all_optimizers_3_functions(seed: int, function: Function, optimizer: Optimizer):
    test_all_optimizers_and_functions(seed, function, optimizer)


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

    design = make_nd_continuous_domain(bounds=np.tile([-1.0, 1.0], (dim, 1)), dimensionality=dim)

    # Sampler
    ran_sampler = RandomUniform(design=design, seed=seed)
    data: ExperimentData = ran_sampler.get_samples(numsamples=numsamples)

    func = function(noise=None, seed=seed, scale_bounds=np.tile([-1.0, 1.0], (dim, 1)), dimensionality=dim)

    # Evaluate the initial samples
    data.fill_output(output=func(data), label="y")

    opt1: Optimizer = optimizer(data=data, seed=seed)

    opt1.iterate(iterations=iterations, function=func)

    data = opt1.extract_data()

    assert data.get_number_of_datapoints() == (iterations + numsamples)


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
