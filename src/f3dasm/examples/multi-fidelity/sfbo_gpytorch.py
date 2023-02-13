import numpy as np
from matplotlib import pyplot as plt

import f3dasm

import botorch
import gpytorch
import torch

### reg parameters

dim = 1
seed = None
noisy_data_bool = 0
numsamples = 100
fun_class = f3dasm.functions.AlpineN2
training_iter = 50
kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel()) \
     #+ gpytorch.kernels.ScaleKernel(gpytorch.kernels.CosineKernel())
# plot_mll = 1
# plot_gpr = 1
# train_surrogate = True
n_test = 500

reg_parameters = f3dasm.regression.gpr.Sogpr_Parameters(
    likelihood=gpytorch.likelihoods.GaussianLikelihood(),
    kernel=kernel,
    noise_fix=1 - noisy_data_bool,
    training_iter=training_iter,
)

###

### opt parameters

iterations = 10
number_of_samples = 5

###

fun = fun_class(
    dimensionality=dim,
    scale_bounds=np.tile([0.0, 1.0], (dim, 1)),
    )

parameter_DesignSpace = f3dasm.make_nd_continuous_design(
    bounds=np.tile([0.0, 1.0], (dim, 1)),
    dimensionality=dim,
)

sampler = f3dasm.sampling.SobolSequence(design=parameter_DesignSpace)

optimizer = f3dasm.optimization.BayesianOptimizationTorch(
    data=f3dasm.Data(design=parameter_DesignSpace),
    )
    
# optimizer.init_parameters()
optimizer.parameter.n_init = number_of_samples
optimizer.parameter.regressor_hyperparameters = reg_parameters
# optimizer.parameter.regressor_hyperparameters.noise_fix = True
# optimizer.parameter.kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.CosineKernel())
# optimizer.parameter.acquisition = f3dasm.base.acquisition.ExpectedImprovement
# optimizer.parameter.acquisition_hyperparameters = f3dasm.optimization.bayesianoptimization_torch.Acquisition_Parameters(
#     best_f=-np.inf,
#     maximize=False,
# )
optimizer.parameter.visualize_gp = True
# optimizer.parameter.acquisition_hyperparameters = {
#     'best_f': np.inf,
#     'maximize': False,
# }

res = f3dasm.run_optimization(
    optimizer=optimizer,
    function=fun,
    sampler=sampler,
    iterations=iterations,
    seed=seed,
    number_of_samples=number_of_samples,
)

res.data['output'] = optimizer.scaler.inverse_transform(res.data['output'])

print(res.data)

# plot_x = np.linspace(fun.input_domain[0, 0], fun.input_domain[0, 1], 500)[:, None]
# plt.plot(plot_x, fun(plot_x))
# plt.scatter(res.data['input'], res.data['output'])

plt.show()