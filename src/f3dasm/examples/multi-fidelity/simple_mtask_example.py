import copy

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

import f3dasm
from f3dasm.base.function import AugmentedFunction
# from f3dasm.functions.adapters.torch_functions import AugmentedTestFunction, botorch_TestFunction
from f3dasm.optimization.bayesianoptimization_torch import mf_data_compiler

dim = 1

base_fun = f3dasm.functions.Sphere(
    dimensionality=dim,
    scale_bounds=np.tile([0.0, 1.0], (dim, 1)),
    )

fids = [0.5, 1.0]
costs = [0.5, 1.0]
samp_nos = [20, 5]

funs = []
mf_design_space = []
mf_sampler = []
mf_train_data = []

for fid_no, (fid, cost, samp_no) in enumerate(zip(fids, costs, samp_nos)):

    fun = AugmentedFunction(
            base_fun=base_fun,
            fid=fid,
            )
    
    parameter_DesignSpace = f3dasm.make_nd_continuous_design(
        bounds=np.tile([0.0, 1.0], (dim, 1)),
        dimensionality=dim,
    )

    fidelity_parameter = f3dasm.ConstantParameter(name="fid", constant_value=np.floor(fid))
    parameter_DesignSpace.add_input_space(fidelity_parameter)

    sampler = f3dasm.sampling.SobolSequence(design=parameter_DesignSpace)

    train_data = sampler.get_samples(numsamples=samp_no)

    train_data.add_output(output=fun(train_data))
    
    # plt.scatter(train_data.data['input', 'x0'], train_data.data['output'])
    # plt.show()

    funs.append(fun)
    mf_design_space.append(parameter_DesignSpace)
    mf_sampler.append(sampler)
    mf_train_data.append(train_data)

mffun = f3dasm.MultiFidelityFunction(
    fidelity_functions=funs,
    fidelity_parameters=fids,
    costs=costs,
)

# train_data: f3dasm.Data = mf_data_compiler(
#     mfdata=mf_train_data,
#     fids=fids,
# )

mf_train_data[-1].data = pd.concat([d.data for d in mf_train_data], ignore_index=True)

regressor = f3dasm.regression.gpr.Mtask(
    mf_train_data=mf_train_data[-1],
    mf_design=mf_train_data[-1].design,
)

surrogate = regressor.train()

test_sampler = f3dasm.sampling.LatinHypercube(design=mf_design_space[-1])
test_data = sampler.get_samples(numsamples=500)

mean, var = surrogate.predict(test_data)

plt.scatter(mf_train_data[-1].data['input', 'x0'], mf_train_data[-1].data['output'])
plt.plot(test_data.data['input', 'x0'], mean, '.')
plt.show()