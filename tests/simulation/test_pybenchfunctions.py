import numpy as np
import pytest
from f3dasm.base.designofexperiments import make_nd_continuous_design
from f3dasm.base.optimization import Optimizer
from f3dasm.base.samplingmethod import SamplingInterface
from f3dasm.optimization.gradient_based_algorithms import SGD, Adam, Momentum
from f3dasm.optimization.pygmo_implementations import CMAES, PSO, SGA, XNES
from f3dasm.sampling.samplers import (
    LatinHypercubeSampling,
    RandomUniformSampling,
    SobolSequenceSampling,
)
from f3dasm.simulation import FUNCTIONS
from f3dasm.optimization import OPTIMIZERS
from f3dasm.simulation.pybenchfunction import PyBenchFunction


# def test_plotting():
#     for _, func in enumerate(FUNCTIONS):
#         if not func.is_dim_compatible(2):
#             continue
#         f = func(dimensionality=2)
#         f.plot(px=10, show=False)


@pytest.mark.parametrize("function", FUNCTIONS)
def test_plotting(function: PyBenchFunction):
    if not function.is_dim_compatible(2):
        return
    f = function(dimensionality=2)
    f.plot(px=10, show=False)


@pytest.mark.parametrize("function", FUNCTIONS)
def test_global_minimum(function: PyBenchFunction):
    dim = 2
    if not function.is_dim_compatible(dim):
        dim = 4
    f = function(dimensionality=dim)
    f.get_global_minimum(dim)


if __name__ == "__main__":  # pragma: no cover
    pytest.main()


@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize("optimizer", OPTIMIZERS)
@pytest.mark.parametrize("function", FUNCTIONS)
def test_adam_on_functions(seed: int, function: PyBenchFunction, optimizer: Optimizer):
    # seed = np.random.randint(low=0, high=10000)
    i = 50  # iterations

    dim = 2
    if not function.is_dim_compatible(dim):
        dim = 4
        if not function.is_dim_compatible(dim):
            dim = 3

    design = make_nd_continuous_design(bounds=[-1.0, 1.0], dimensions=dim)

    # Sampler
    ran_sampler = RandomUniformSampling(doe=design, seed=seed)
    data = ran_sampler.get_samples(numsamples=30)

    func = function(noise=False, seed=42, dimensionality=dim)

    # Evaluate the initial samples
    data.add_output(output=func.eval(data), label="y")

    opt1 = optimizer(data=data, seed=seed)
    opt2 = optimizer(data=data, seed=seed)

    opt1.iterate(iterations=i, function=func)
    opt2.iterate(iterations=i, function=func)
    data = opt1.extract_data()
    data2 = opt2.extract_data()

    assert all(data.data == data2.data)
