import numpy as np
from matplotlib import pyplot as plt

import f3dasm
from f3dasm.design import ExperimentData

import botorch
import gpytorch
import torch

from f3dasm.base.function import AugmentedFunction, MultiFidelityFunction

### reg parameters

dim = 1
seed = 123
noisy_data_bool = 1
numsamples = 100
fun_class = f3dasm.functions.AlpineN2
training_iter = 50
kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel()) \
     #+ gpytorch.kernels.ScaleKernel(gpytorch.kernels.CosineKernel())
# plot_mll = 1
# plot_gpr = 1
# train_surrogate = True
n_test = 500
likelihood = gpytorch.likelihoods.GaussianLikelihood()
opt_algo = torch.optim.Adam
opt_algo_kwargs = dict(lr=0.1)

base_fun = fun_class(
    dimensionality=dim,
    scale_bounds=np.tile([0.0, 1.0], (dim, 1)),
    )

mean_module_list = torch.nn.ModuleList([
    gpytorch.means.ZeroMean(),
    gpytorch.means.ZeroMean()
])

covar_module_list = torch.nn.ModuleList([
    gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
    gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
])

reg_parameters = f3dasm.machinelearning.gpr.Cokgj_Parameters(
    likelihood=likelihood,
    kernel=covar_module_list,
    mean=mean_module_list,
    noise_fix=1- noisy_data_bool,
    opt_algo=opt_algo,
    opt_algo_kwargs=opt_algo_kwargs,
    verbose_training=0,
    training_iter=training_iter,
    )

### opt parameters

iterations = 10
numbers_of_samples = [20, 5]
fidelity_parameters = [0.5, 1.]
costs = [0.5, 1.]
budget = 10

###

fidelity_functions = []
multifidelity_samplers = []

for i, fidelity_parameter in enumerate(fidelity_parameters):

    fun = AugmentedFunction(
            base_fun=base_fun,
            fid=fidelity_parameter,
            dimensionality=base_fun.dimensionality,
            scale_bounds=base_fun.scale_bounds,
            )
    
    parameter_DesignSpace = f3dasm.make_nd_continuous_design(
        bounds=np.tile([0.0, 1.0], (dim, 1)),
        dimensionality=dim,
    )
    # fidelity_parameter = f3dasm.ConstantParameter(name="fid", constant_value=fidelity_parameter)
    # parameter_DesignSpace.add_input_space(fidelity_parameter)

    sampler = f3dasm.sampling.SobolSequence(design=parameter_DesignSpace, seed=seed)

    fidelity_functions.append(fun)
    multifidelity_samplers.append(sampler)

multifidelity_function = MultiFidelityFunction(
    fidelity_functions=fidelity_functions,
    fidelity_parameters=fidelity_parameters,
    costs=costs,
)

optimizer = f3dasm.optimization.MFBayesianOptimizationTorch(
    data=ExperimentData(design=parameter_DesignSpace),
    multifidelity_function=multifidelity_function,
)

# optimizer.init_parameters()
optimizer.parameter.n_init = numbers_of_samples
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

res = f3dasm.run_multi_fidelity_optimization(
    optimizer=optimizer,
    multifidelity_function=multifidelity_function,
    multifidelity_samplers=multifidelity_samplers,
    iterations=iterations,
    seed=seed,
    numbers_of_samples=numbers_of_samples,
    budget=budget
)

print(res[0].data)
print()
print(res[1].data)