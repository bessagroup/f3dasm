import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import f3dasm
from f3dasm import AugmentedFunction, Data
from f3dasm.base import function

dim = 1
iterations = 40
seed = 123
number_of_samples = 10
samp_nos = [20, 10]

fids = [0.5, 1.]
costs = [0.5, 1.]

base_fun = f3dasm.functions.Schwefel(
    dimensionality=dim,
    scale_bounds=np.tile([0.0, 1.0], (dim, 1)),
    )

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
        bounds=base_fun.input_domain.astype(float),
        dimensionality=dim,
    )
    fidelity_parameter = f3dasm.ConstantParameter(name="fid", constant_value=fid)
    parameter_DesignSpace.add_input_space(fidelity_parameter)

    sampler = f3dasm.sampling.SobolSequenceSampling(design=parameter_DesignSpace)

    init_train_data = sampler.get_samples(numsamples=samp_no)
    init_train_data.add_output(output=fun(init_train_data))

    funs.append(fun)
    mf_design_space.append(parameter_DesignSpace)
    mf_sampler.append(sampler)
    mf_train_data.append(init_train_data)

mffun = function.MultiFidelityFunction(
    funs=funs,
    fids=fids,
    costs=costs,
)

optimizer = f3dasm.optimization.MFBayesianOptimizationTorch(
    data=mf_train_data,
    mffun=mffun,
)

optimizer.init_parameters()
print(optimizer)

res = f3dasm.run_mf_optimization(
    optimizer=optimizer,
    mffunction=mffun,
    sampler=mf_sampler[-1],
    iterations=10,
    seed=123,
    number_of_samples=samp_nos,
    # budget=10 # add budget to optimization iterator
)

print(res)

# plot_x = np.linspace(fun.base_fun.input_domain[0, 0], fun.base_fun.input_domain[0, 1], 500)[:, None]
# plt.plot(plot_x, fun(np.hstack((np.ones_like(plot_x), plot_x))))
# plt.scatter(res[1].data['input', 'x0'], res[1].data['output'])
# plt.show()