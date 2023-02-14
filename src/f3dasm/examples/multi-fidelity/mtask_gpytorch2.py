import f3dasm
import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from typing import Any

from sklearn.preprocessing import StandardScaler

from f3dasm.base.function import AugmentedFunction

###

dim = 1
seed = 123
noisy_data_bool = 1
fun_class = f3dasm.functions.Sphere
opt_algo = torch.optim.Adam
opt_algo_kwargs = dict(lr=0.1)
training_iter = 150
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

mean_module = gpytorch.means.ZeroMean()
covar_module = gpytorch.kernels.RBFKernel()

###

class Forrester(f3dasm.functions.Function):
    def __init__(self, dimensionality: int, seed: Any or int = None):
        super().__init__(dimensionality, seed)

    def f(self, x):
        return (6 * x - 2) ** 2 * np.sin(12 * x - 4)

class Forrester_lf(f3dasm.functions.Function):
    def __init__(self, dimensionality: int, seed: Any or int = None):
        super().__init__(dimensionality, seed)

    def f(self, x):
        return 0.5 * (6 * x - 2) ** 2 * np.sin(12 * x - 4) + 10 * (x - 0.5) - 5

base_fun = fun_class(
    dimensionality=dim,
    scale_bounds=np.tile([0.0, 1.0], (dim, 1)),
    )

fids = [0.5, 1.0]
costs = [0.5, 1.0]
samp_nos = [11, 100]

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
        fun = Forrester_lf(dimensionality=1)
    else:
        fun = Forrester(dimensionality=1)
    
    parameter_DesignSpace = f3dasm.make_nd_continuous_design(
        bounds=np.tile([0.0, 1.0], (dim, 1)),
        dimensionality=dim,
    )

    sampler = f3dasm.sampling.SobolSequence(design=parameter_DesignSpace, seed=seed)

    if train_data_supplied:
        train_data = f3dasm.Data(design=parameter_DesignSpace)

        if fid_no == 0:
            input_array = np.linspace(0, 1, 11)[:, None]
        else:
            input_array = np.array([0., 0.4, 0.6, 1.])[:, None]

        train_data.add_numpy_arrays(
            input=input_array, 
            output=np.full_like(input_array, np.nan)
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

    funs.append(fun)
    mf_design_space.append(parameter_DesignSpace)
    mf_sampler.append(sampler)
    mf_train_data.append(train_data)

###

param = f3dasm.regression.gpr.Stmf_Parameters(
    likelihood=likelihood,
    kernel=covar_module,
    mean=mean_module,
    noise_fix=1- noisy_data_bool,
    opt_algo=opt_algo,
    opt_algo_kwargs=opt_algo_kwargs,
    verbose_training=1,
    training_iter=training_iter,
    )

regressor = f3dasm.regression.gpr.Stmf(
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

# # Test points are regularly spaced along [0,1]
# # Make predictions by feeding model through likelihood
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_sampler = f3dasm.sampling.SobolSequence(design=parameter_DesignSpace, seed=0)
    
    if dim == 1:
        test_x = torch.linspace(0, 1, n_test)[:, None]
    else:
        test_x = torch.tensor(test_sampler.get_samples(numsamples=n_test).get_input_data().values)
   
    observed_pred = surrogate.predict([torch.tensor([])[:, None], test_x])
    # observed_pred = surrogate.predict([test_x, torch.tensor([])[:, None]])
    # observed_pred = surrogate.predict(test_x)
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
        # train_x=torch.tensor(mf_train_data[0].data.input.x0.values),
        # train_y=torch.tensor(scaler.inverse_transform(mf_train_data[0].data.output.y.values[:, None])),
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