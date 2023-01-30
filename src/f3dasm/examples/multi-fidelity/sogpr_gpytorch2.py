import f3dasm
import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

###

dim = 1
seed = 123
noisy_data_bool = 1
numsamples = 100
fun_class = f3dasm.functions.Sphere
opt_algo = torch.optim.Adam
training_iter = 50
mean = gpytorch.means.ZeroMean()
kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel()) \
     #+ gpytorch.kernels.ScaleKernel(gpytorch.kernels.CosineKernel())
# plot_mll = 1
# plot_gpr = 1
# train_surrogate = True
n_test = 500

###

fun = fun_class(
    dimensionality=dim,
    scale_bounds=np.tile([0.0, 1.0], (dim, 1)),
    )

parameter_DesignSpace: f3dasm.DesignSpace = f3dasm.make_nd_continuous_design(
    bounds=np.tile([0.0, 1.0], (dim, 1)),
    dimensionality=dim,
)

sampler = f3dasm.sampling.SobolSequence(design=parameter_DesignSpace, seed=seed)

train_data: f3dasm.Data = sampler.get_samples(numsamples=numsamples)
# opt_retries = 50

output = fun(train_data) 
# train_data.add_output(output=output)

train_x = torch.tensor(train_data.get_input_data().values)
train_y = torch.tensor(train_data.get_output_data().values.flatten())

scaler = StandardScaler()
scaler.fit(train_y.numpy()[:, None])
train_y_scaled = torch.tensor(scaler.transform(train_y.numpy()[:, None]).flatten())

train_y_scaled += noisy_data_bool * np.random.randn(*train_y_scaled.shape) * math.sqrt(0.04)

train_data.add_output(output=train_y_scaled)

###

param = f3dasm.regression.gpr.Sogpr_Parameters(
    kernel=kernel,
    mean=mean,
    noise_fix=1- noisy_data_bool,
    opt_algo=opt_algo,
    verbose_training=False,
    training_iter=training_iter,
    )

regressor = f3dasm.regression.gpr.Sogpr(
    train_data=train_data, 
    design=train_data.design,
    parameter=param,
    # noise_fix=1 - noisy_data_bool,
)

surrogate = regressor.train(
    # opt_algo=opt_algo,
    # verbose_optimization=False,
    # training_iter=training_iter,
)

###

# Get into evaluation (predictive posterior) mode
surrogate.model.eval()
surrogate.model.likelihood.eval()

# # Test points are regularly spaced along [0,1]
# # Make predictions by feeding model through likelihood
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_sampler = f3dasm.sampling.SobolSequence(design=parameter_DesignSpace, seed=0)
    
    if dim == 1:
        test_x = torch.linspace(0, 1, n_test)
    else:
        test_x = torch.tensor(test_sampler.get_samples(numsamples=n_test).get_input_data().values)
    
    # observed_pred = surrogate.model.likelihood(surrogate.model(test_x))
    observed_pred = surrogate.predict(test_x)
    exact_y = fun(test_x.numpy()[:, None])

###

if dim == 1:
    surrogate.plot_gpr(
        test_x=test_x, 
        scaler=scaler, 
        exact_y=exact_y, 
        observed_pred=observed_pred,
        train_x=train_x,
        train_y=torch.tensor(scaler.inverse_transform(train_y_scaled.numpy()[:, None]))
        )

surrogate.plot_mll(
    train_x=train_x,
    train_y_scaled=train_y_scaled,
)

plt.show()