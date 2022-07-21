from typing import Tuple
import pytest
from f3dasm.base.data import Data

from f3dasm.base.designofexperiments import make_nd_continuous_design
from f3dasm.base.optimization import Optimizer
from f3dasm.base.simulation import Function
from f3dasm.optimization.pygmo_implementations import CMAES, PSO, XNES, SGA
from f3dasm.optimization.gradient_based_algorithms import SGD, Adam, Momentum
from f3dasm.optimization.gpyopt_implementations import BayesianOptimization
from f3dasm.sampling.samplers import RandomUniformSampling
from f3dasm.simulation.benchmark_functions import Levy


@pytest.fixture
def data_seed():
    seed = 42
    design = make_nd_continuous_design(bounds=[-1.0, 1.0], dimensions=10)

    # Sampler
    ran_sampler = RandomUniformSampling(doe=design, seed=seed)
    data = ran_sampler.get_samples(numsamples=30)

    func = Levy(noise=False, seed=42)

    # Evaluate the initial samples
    data.add_output(output=func.eval(data), label="y")

    return data, seed, func


@pytest.mark.parametrize(
    "optimizer",
    [CMAES, PSO, XNES, SGA, SGD, Adam, Momentum],
)
def test_all_optimizers(data_seed: Tuple[Data, int, Function], optimizer):
    data, seed, func = data_seed

    opt1 = optimizer(data=data, seed=seed)
    opt2 = optimizer(data=data, seed=seed)

    i = 10
    opt1.iterate(iterations=i, function=func)
    opt2.iterate(iterations=i, function=func)
    data = opt1.extract_data()
    data2 = opt2.extract_data()

    assert all(data.data == data2.data)
