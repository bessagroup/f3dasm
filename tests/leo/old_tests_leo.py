import pytest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import f3dasm
from f3dasm.base.function import AugmentedFunction, MultiFidelityFunction
from f3dasm.functions import FUNCTIONS, get_functions

import gpytorch
import torch
import math
from sklearn.preprocessing import StandardScaler

# pytestmark = pytest.mark.leo

# @pytest.mark.parametrize("dim", [1, 2, 3])
# @pytest.mark.parametrize("function", FUNCTIONS)
# def test_gpr(dim: int, function: f3dasm.Function):
#     if not function.is_dim_compatible(dim):
#         return

#     fun = function(
#         dimensionality=dim,
#         scale_bounds=np.tile([0.0, 1.0], (dim, 1)),
#     )

#     parameter_DesignSpace: f3dasm.DesignSpace = f3dasm.make_nd_continuous_design(
#         bounds=np.tile([0.0, 1.0], (dim, 1)),
#         dimensionality=dim,
#     )

#     sampler = f3dasm.sampling.SobolSequence(design=parameter_DesignSpace)

#     train_data: f3dasm.Data = sampler.get_samples(numsamples=8)

#     train_data.add_output(output=fun(train_data))

#     regressor = f3dasm.regression.gpr.Sogpr(
#         train_data=train_data, 
#         design=train_data.design,
#     )

#     surrogate = regressor.train()

#     test_input_data: f3dasm.Data = sampler.get_samples(numsamples=500)
#     mean, var = surrogate.predict(test_input_data=test_input_data)


# @pytest.mark.parametrize("dim", [1, 2, 3])
# def test_bo(dim: int):
#     # dim = 1
#     iterations = 40
#     seed = 123
#     number_of_samples = 10

#     fun = f3dasm.functions.Schwefel(
#         dimensionality=dim,
#         scale_bounds=np.tile([0.0, 1.0], (dim, 1)),
#         )
#     parameter_DesignSpace = f3dasm.make_nd_continuous_design(
#         bounds=np.tile([0.0, 1.0], (dim, 1)),
#         dimensionality=dim,
#     )
#     SobolSampler = f3dasm.sampling.SobolSequence(design=parameter_DesignSpace)
#     samples = SobolSampler.get_samples(numsamples=8)
#     samples.add_output(output=fun(samples))
#     optimizer = f3dasm.optimization.BayesianOptimizationTorch(data=samples)
#     optimizer.init_parameters()
#     res = f3dasm.run_optimization(
#         optimizer=optimizer,
#         function=fun,
#         sampler=SobolSampler,
#         iterations=iterations,
#         seed=seed,
#         number_of_samples=number_of_samples,
#     )


# @pytest.mark.parametrize("dim", [1, 2, 3])
# @pytest.mark.parametrize("function", FUNCTIONS)
# def test_mf_gpr(dim: int, function: f3dasm.Function):
#     if not function.is_dim_compatible(dim):
#         return

#     base_fun = function(
#         dimensionality=dim,
#         scale_bounds=np.tile([0.0, 1.0], (dim, 1)),
#     )

#     fids = [0.5, 1.0]
#     costs = [0.5, 1.0]
#     samp_nos = [20, 10]

#     funs = []
#     mf_design_space = []
#     mf_sampler = []
#     mf_train_data = []

#     for fid_no, (fid, cost, samp_no) in enumerate(zip(fids, costs, samp_nos)):

#         fun = AugmentedFunction(
#                 base_fun=base_fun,
#                 fid=fid,
#                 )
        
#         parameter_DesignSpace = f3dasm.make_nd_continuous_design(
#             bounds=np.tile([0.0, 1.0], (dim, 1)),
#             dimensionality=dim,
#         )

#         fidelity_parameter = f3dasm.ConstantParameter(name="fid", constant_value=fid)
#         parameter_DesignSpace.add_input_space(fidelity_parameter)

#         sampler = f3dasm.sampling.SobolSequence(design=parameter_DesignSpace)

#         train_data = sampler.get_samples(numsamples=samp_no)

#         train_data.add_output(output=fun(train_data))

#         funs.append(fun)
#         mf_design_space.append(parameter_DesignSpace)
#         mf_sampler.append(sampler)
#         mf_train_data.append(train_data)

#     # mffun = f3dasm.base.function.MultiFidelityFunction(
#     #     funs=funs,
#     #     fids=fids,
#     #     costs=costs,
#     # )

#     mf_train_data[-1].data = pd.concat([d.data for d in mf_train_data], ignore_index=True)

#     regressor = f3dasm.regression.gpr.Stmf(
#         mf_train_data=mf_train_data[-1],
#         mf_design=mf_train_data[-1].design,
#     )

#     surrogate = regressor.train()

#     test_sampler = f3dasm.sampling.LatinHypercube(design=mf_design_space[-1])
#     test_data = test_sampler.get_samples(numsamples=500)

#     mean, var = surrogate.predict(test_data)


# @pytest.mark.parametrize("dim", [1, 2, 3])
# def test_mf_bo(dim: int):
#     # dim = 1
#     iterations = 10
#     seed = 123
#     samp_nos = [20, 10]

#     fids = [0.5, 1.]
#     costs = [0.5, 1.]

#     base_fun = f3dasm.functions.Schwefel(
#         dimensionality=dim,
#         scale_bounds=np.tile([0.0, 1.0], (dim, 1)),
#         )

#     funs = []
#     mf_design_space = []
#     mf_sampler = []
#     mf_train_data = []

#     for fid_no, (fid, cost, samp_no) in enumerate(zip(fids, costs, samp_nos)):

#         fun = AugmentedFunction(
#                 base_fun=base_fun,
#                 fid=fid,
#                 )
        
#         parameter_DesignSpace = f3dasm.make_nd_continuous_design(
#             bounds=np.tile([0.0, 1.0], (dim, 1)),
#             dimensionality=dim,
#         )
#         fidelity_parameter = f3dasm.ConstantParameter(name="fid", constant_value=fid)
#         parameter_DesignSpace.add_input_space(fidelity_parameter)

#         sampler = f3dasm.sampling.SobolSequence(design=parameter_DesignSpace)

#         init_train_data = sampler.get_samples(numsamples=samp_no)
#         init_train_data.add_output(output=fun(init_train_data))

#         funs.append(fun)
#         mf_design_space.append(parameter_DesignSpace)
#         mf_sampler.append(sampler)
#         mf_train_data.append(init_train_data)

#     mffun = MultiFidelityFunction(
#         funs=funs,
#         fids=fids,
#         costs=costs,
#     )

#     optimizer = f3dasm.optimization.MFBayesianOptimizationTorch(
#         data=mf_train_data,
#         mffun=mffun,
#     )

#     optimizer.init_parameters()

#     res = f3dasm.run_mf_optimization(
#         optimizer=optimizer,
#         mffunction=mffun,
#         sampler=mf_sampler[-1],
#         iterations=iterations,
#         seed=123,
#         number_of_samples=samp_nos,
#         # budget=10 # add budget to optimization iterator
#     )