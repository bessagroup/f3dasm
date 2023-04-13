import numpy as np
from matplotlib import pyplot as plt

import f3dasm
from f3dasm.design import ExperimentData

import gpytorch
import torch

from f3dasm.functions import MultiFidelityFunction
from f3dasm.functions.fidelity_augmentors import NoiseInterpolator, NoisyBaseFun

### reg parameters

dim = 1
seed = 123
noisy_data_bool = 1
numsamples = 100
fun_class = f3dasm.functions.AlpineN2
training_iter = 50
n_test = 500
likelihood = gpytorch.likelihoods.GaussianLikelihood()
opt_algo = torch.optim.Adam
opt_algo_kwargs = dict(lr=0.1)

base_fun = fun_class(
    dimensionality=dim,
    scale_bounds=np.tile([0.0, 1.0], (dim, 1)),
    offset=False,
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

iterations = 1
numbers_of_samples = [20, 5]
aug_coeffs = [0.5, 1.]
costs = [0.5, 1.]
budget = 10

###

fidelity_functions = []
multifidelity_samplers = []

for i, aug_coeff in enumerate(aug_coeffs):

    noise_augmentor = NoiseInterpolator(noise_var=1., aug_coeff=aug_coeff)

    if i == 0:
        fidelity_function = NoisyBaseFun(
            seed=seed,
            base_fun=base_fun, 
            dimensionality=dim,
            noise_augmentor=noise_augmentor, 
            )                
        
        # fidelity_function = BiasedBaseFun(
        #     seed=seed,
        #     base_fun=fun, 
        #     dimensionality=config.dimensionality,
        #     scale_augmentor=scale_augmentor, 
        #     )

    else:
        fidelity_function = base_fun
    
    parameter_DesignSpace = f3dasm.make_nd_continuous_design(
        bounds=np.tile([0.0, 1.0], (dim, 1)),
        dimensionality=dim,
    )
    # fidelity_parameter = f3dasm.ConstantParameter(name="fid", constant_value=fidelity_parameter)
    # parameter_DesignSpace.add_input_space(fidelity_parameter)

    sampler = f3dasm.sampling.SobolSequence(design=parameter_DesignSpace, seed=seed)

    fidelity_functions.append(fidelity_function)
    multifidelity_samplers.append(sampler)

multifidelity_function = MultiFidelityFunction(
    fidelity_functions=fidelity_functions,
    fidelity_parameters=aug_coeffs,
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
    function=multifidelity_function,
    sampler=multifidelity_samplers,
    iterations=iterations,
    seed=seed,
    number_of_samples=numbers_of_samples,
    budget=budget
)

print(res[0].data)
print()
print(res[1].data)

plt.show()