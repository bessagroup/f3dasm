import numpy as np
import torch
from matplotlib import pyplot as plt

import f3dasm
from f3dasm.base.function import AugmentedFunction

from f3dasm.functions.adapters.torch_functions import AugmentedTestFunction, botorch_TestFunction
from f3dasm.regression.gpr import Cokgj, Mtask, Stmf

dim = 2

base_fun = f3dasm.functions.Ackley(dimensionality=dim)
fids = [0.5, 1]
costs = [0.5, 1]
samp_nos = [20, 10]
funs = []
for fid_no, fid in enumerate(fids):
    funs.append(
        AugmentedFunction(
            base_fun=base_fun,
            fid=fid,
        )
    )

mffun = f3dasm.base.function.MultiFidelityFunction(
    funs=funs,
    fids=fids,
    costs=costs,
)

parameter_DesignSpace = f3dasm.make_nd_continuous_design(
    bounds=base_fun.input_domain.astype(float),
    dimensionality=dim,
)

SobolSampler = f3dasm.sampling.SobolSequenceSampling(design=parameter_DesignSpace)

mfsamples = []
for fid_no, samp_no in enumerate(samp_nos):
    samples = SobolSampler.get_samples(numsamples=samp_no)

    samples.add_output(output=mffun.funs[fid_no](samples))

    mfsamples.append(samples)

print(mfsamples)

# dim = 1

# fun = f3dasm.functions.Schwefel(dimensionality=dim)
# parameter_DesignSpace = f3dasm.make_nd_continuous_design(
#     bounds=fun.input_domain.astype(float),
#     dimensionality=dim,
# )

# SobolSampler = f3dasm.sampling.SobolSequenceSampling(design=parameter_DesignSpace)

# aug_fun = AugmentedTestFunction(botorch_TestFunction(fun=fun), noise_type="b")

# train_x_hf_space = SobolSampler.sample_continuous(numsamples=5)
# train_x_lf_space = SobolSampler.sample_continuous(numsamples=200)
# train_x_lf_fid = 0.5 * np.ones_like(train_x_lf_space)
# train_x_lf = np.hstack((train_x_lf_space, train_x_lf_fid))
# train_x_hf_fid = np.ones_like(train_x_hf_space)
# train_x_hf = np.hstack((train_x_hf_space, train_x_hf_fid))
# train_x_mf = torch.tensor(np.vstack((train_x_hf, train_x_lf)))
# train_y_mf = aug_fun(train_x_mf)[:, None]

# fidelity_parameter = f3dasm.ContinuousParameter(name="fid", lower_bound=0.0, upper_bound=1.0)
# parameter_DesignSpace.add_input_space(fidelity_parameter)

# mf_regressor = Stmf(mf_train_input_data=train_x_mf, mf_train_output_data=train_y_mf, mf_design=parameter_DesignSpace)
# mf_surrogate = mf_regressor.train()

# test_x_hf_space = np.linspace(fun.input_domain[0, 0], fun.input_domain[0, 1], 500)[:, None]
# test_x_hf_fid = np.ones_like(test_x_hf_space)
# test_x_hf = np.hstack((test_x_hf_space, test_x_hf_fid))
# # mean_hf, var_hf = mf_surrogate.predict(test_x_hf_space)
# mean_hf, var_hf = mf_surrogate.predict(test_x_hf)
# # mean_hf = mean_hf[:, 1][:, None]
# # var_hf = var_hf[:500]
# ucb, lcb = [mean_hf + (-1) ** k * 2 * np.sqrt(np.abs(var_hf)) for k in range(2)]

# plt.plot(test_x_hf_space, fun(test_x_hf_space))
# plt.plot(test_x_hf_space, mean_hf, "r--")
# plt.plot(test_x_hf_space, lcb, "k", linewidth=0.5)
# plt.plot(test_x_hf_space, ucb, "k", linewidth=0.5)
# plt.fill_between(test_x_hf_space.flatten(), lcb.flatten(), ucb.flatten(), color="r", alpha=0.1)
# plt.scatter(train_x_hf_space, train_y_mf[: len(train_x_hf_space)], c="r")
# plt.grid()
# plt.tight_layout()

# plt.show()