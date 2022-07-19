# from typing import Tuple
# import pytest

# from f3dasm.base.designofexperiments import make_nd_continuous_design
# from f3dasm.base.optimization import Optimizer
# from f3dasm.base.simulation import Function
# from f3dasm.optimization.gradient_based_algorithms import SGD
# from f3dasm.sampling.samplers import RandomUniformSampling
# from f3dasm.simulation.benchmark_functions import Levy


# @pytest.fixture
# def optimizer_and_function():
#     seed = 42
#     hyperparameters = {"step_size": 1e-3}
#     design = make_nd_continuous_design(bounds=[-1.0, 1.0], dimensions=10)

#     # Sampler
#     ran_sampler = RandomUniformSampling(doe=design, seed=seed)
#     data = ran_sampler.get_samples(numsamples=1)

#     levy = Levy(noise=False, seed=42)

#     # Evaluate the initial samples
#     data.add_output(output=levy.eval(data), label="y")

#     # algorithm
#     sgd = SGD(data=data, hyperparameters=hyperparameters, seed=seed)
#     return sgd


# def test_sgd(optimizer_and_function: Optimizer):
#     sgd = optimizer_and_function
#     levy = Levy(noise=False, seed=42)
#     i = 100
#     sgd.iterate(iterations=i, function=levy)
#     data = sgd.extract_data()
#     print(data)
