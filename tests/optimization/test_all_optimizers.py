import pytest
from f3dasm.base.utils import make_nd_continuous_design
from f3dasm.base.optimization import Optimizer
from f3dasm.base.function import Function
from f3dasm.sampling.samplers import (
    RandomUniformSampling,
)
from f3dasm.functions import FUNCTIONS, FUNCTIONS_2D
from f3dasm.optimization import OPTIMIZERS


@pytest.mark.parametrize("function", FUNCTIONS_2D)
def test_plotting(function: Function):
    f = function(dimensionality=2)
    f.plot(px=10, show=False)


@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize("optimizer", OPTIMIZERS)
@pytest.mark.parametrize("function", FUNCTIONS)
def test_all_optimizers_and_functions(seed: int, function: Function, optimizer: Optimizer):
    i = 50  # iterations

    dim = 6
    if not function.is_dim_compatible(dim):
        dim = 4
        if not function.is_dim_compatible(dim):
            dim = 3
            if not function.is_dim_compatible(dim):
                dim = 2

    design = make_nd_continuous_design(bounds=[-1.0, 1.0], dimensions=dim)

    # Sampler
    ran_sampler = RandomUniformSampling(doe=design, seed=seed)
    data = ran_sampler.get_samples(numsamples=30)

    func = function(noise=False, seed=seed, scale_bounds=[-1.0, 1.0], dimensionality=dim)

    # Evaluate the initial samples
    data.add_output(output=func.eval(data), label="y")

    opt1 = optimizer(data=data, seed=seed)
    opt2 = optimizer(data=data, seed=seed)

    opt1.iterate(iterations=i, function=func)
    opt2.iterate(iterations=i, function=func)
    data = opt1.extract_data()
    data2 = opt2.extract_data()

    assert all(data.data == data2.data)


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
