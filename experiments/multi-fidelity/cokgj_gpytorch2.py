import f3dasm
import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from typing import Any

from sklearn.preprocessing import StandardScaler

from f3dasm.functions import AugmentedFunction

from f3dasm.design import ExperimentData

###

dim = 1
seed = 123
noisy_data_bool = 1
# numsamples = 10
# fun_class = f3dasm.functions.Sphere
opt_algo = torch.optim.Adam
opt_algo_kwargs = dict(lr=0.1)
training_iter = 150
# mean = gpytorch.means.ZeroMean()
# kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()) \
     #+ gpytorch.kernels.ScaleKernel(gpytorch.kernels.CosineKernel())
# plot_mll = 1
# plot_gpr = 1
# train_surrogate = True
n_test = 500
likelihood = gpytorch.likelihoods.GaussianLikelihood()
train_data_supplied = 1

mean_module_list = torch.nn.ModuleList([
    gpytorch.means.ZeroMean(),
    gpytorch.means.ZeroMean()
])

covar_module_list = torch.nn.ModuleList([
    gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
    gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
])

###

# fun = fun_class(
#     dimensionality=dim,
#     scale_bounds=np.tile([0.0, 1.0], (dim, 1)),
#     )

# parameter_DesignSpace: f3dasm.DesignSpace = f3dasm.make_nd_continuous_design(
#     bounds=np.tile([0.0, 1.0], (dim, 1)),
#     dimensionality=dim,
# )

# sampler = f3dasm.sampling.SobolSequence(design=parameter_DesignSpace, seed=seed)

# train_data: f3dasm.Data = sampler.get_samples(numsamples=numsamples)

# output = fun(train_data) 

# train_x = torch.tensor(train_data.get_input_data().values)
# train_y = torch.tensor(train_data.get_output_data().values.flatten())

# # Scaling the output
# scaler = StandardScaler()
# scaler.fit(train_y.numpy()[:, None])
# train_y_scaled = torch.tensor(scaler.transform(train_y.numpy()[:, None]).flatten())

# train_y_scaled += noisy_data_bool * np.random.randn(*train_y_scaled.shape) * math.sqrt(0.04)

# train_data.add_output(output=train_y_scaled)

###

class Forrester(f3dasm.functions.Function):
    def __init__(self, seed: Any or int = None):
        super().__init__(seed)

    def evaluate(self, x):
        return (6 * x - 2) ** 2 * np.sin(12 * x - 4)

class Forrester_lf(f3dasm.functions.Function):
    def __init__(self, seed: Any or int = None):
        super().__init__(seed)

    def evaluate(self, x):
        return 0.5 * (6 * x - 2) ** 2 * np.sin(12 * x - 4) + 10 * (x - 0.5) - 5

base_fun = f3dasm.functions.AlpineN2(
    dimensionality=dim,
    scale_bounds=np.tile([0.0, 1.0], (dim, 1)),
    )

# base_fun = Forrester(dimensionality=1)

fids = [0.5, 1.0]
costs = [0.5, 1.0]
samp_nos = [11, 4]

funs = []
mf_design_space = []
mf_sampler = []
mf_train_data = []

for fid_no, (fid, cost, samp_no) in enumerate(zip(fids, costs, samp_nos)):

    # fun = AugmentedFunction(
    #         base_fun=base_fun,
    #         fid=fid,
    #         )

    if fid_no == 0:
        fun = Forrester_lf()
    else:
        fun = Forrester()
    
    parameter_DesignSpace = f3dasm.make_nd_continuous_design(
        bounds=np.tile([0.0, 1.0], (dim, 1)),
        dimensionality=dim,
    )

    # fidelity_parameter = f3dasm.ConstantParameter(name="fid", constant_value=fid)
    # parameter_DesignSpace.add_input_space(fidelity_parameter)

    sampler = f3dasm.sampling.SobolSequence(design=parameter_DesignSpace, seed=seed)

    if train_data_supplied:
        train_data = ExperimentData(design=parameter_DesignSpace)

        if fid_no == 1:
            input_array = np.array([0., 0.4, 0.6, 1.])[:, None]
        else:
            input_array = np.linspace(0, 1, 11)[:, None]

        train_data.add_numpy_arrays(
            input_rows=input_array, 
            output_rows=np.full_like(input_array, np.nan)
            )
    else:
        train_data = sampler.get_samples(numsamples=samp_no)

    output = fun(train_data) 

    train_x = torch.tensor(train_data.get_input_data().values)
    train_y = torch.tensor(train_data.get_output_data().values.flatten())

    # Scaling the output
    if fid_no == 0:
        scaler = StandardScaler()
        scaler.fit(train_y.numpy()[:, None])
    train_y_scaled = torch.tensor(scaler.transform(train_y.numpy()[:, None]).flatten())

    # if fid_no == 0:
    #     train_y_scaled += noisy_data_bool * np.random.randn(*train_y_scaled.shape) * math.sqrt(0.04)

    train_data.add_output(output=train_y_scaled)

    # train_data.add_output(output=fun(train_data))

    funs.append(fun)
    mf_design_space.append(parameter_DesignSpace)
    mf_sampler.append(sampler)
    mf_train_data.append(train_data)

###

param = f3dasm.machinelearning.gpr.Cokgj_Parameters(
    likelihood=likelihood,
    kernel=covar_module_list,
    mean=mean_module_list,
    noise_fix=1- noisy_data_bool,
    opt_algo=opt_algo,
    opt_algo_kwargs=opt_algo_kwargs,
    verbose_training=1,
    training_iter=training_iter,
    )

regressor = f3dasm.machinelearning.gpr.Cokgj(
    mf_train_data=mf_train_data, 
    design=train_data.design,
    parameter=param,
)

surrogate = regressor.train()

print()
print(dict(surrogate.model.named_parameters()))

###

# Get into evaluation (predictive posterior) mode
surrogate.model.eval()
surrogate.model.likelihood.eval()

# surrogate.model.train_x = torch.cat(surrogate.model.train_x)
# surrogate.model.train_y = torch.cat(surrogate.model.train_y)

# # Test points are regularly spaced along [0,1]
# # Make predictions by feeding model through likelihood
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_sampler = f3dasm.sampling.SobolSequence(design=parameter_DesignSpace, seed=0)
    
    if dim == 1:
        test_x = torch.linspace(0, 1, n_test)[:, None]
    else:
        test_x = torch.tensor(test_sampler.get_samples(numsamples=n_test).get_input_data().values)
   
    # observed_pred = surrogate.predict([test_x, test_x])
    # observed_pred = surrogate.predict([torch.tensor([])[:, None], test_x])
    observed_pred = surrogate.predict([torch.empty(0, dim), test_x])
    # observed_pred = surrogate.predict([test_x, torch.tensor([])[:, None]])
    exact_y = fun(test_x.numpy())#[:, None])

###

if dim == 1:
    surrogate.plot_gpr(
        test_x=test_x.flatten(), 
        scaler=scaler, 
        exact_y=exact_y, 
        observed_pred=observed_pred,
        train_x=train_x,
        train_y=torch.tensor(scaler.inverse_transform(train_y_scaled.numpy()[:, None]))
        )

# surrogate.plot_mll(
#     train_x=train_x,
#     train_y_scaled=train_y_scaled,
# )

metrics_df = surrogate.gp_metrics(
    scaler=scaler,
    observed_pred=observed_pred,
    exact_y=exact_y.flatten(),
)

print()
print(metrics_df)

plt.show()