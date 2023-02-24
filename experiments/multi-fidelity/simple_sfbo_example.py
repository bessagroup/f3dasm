import numpy as np
from matplotlib import pyplot as plt

import f3dasm

import botorch
import gpytorch

dim = 1
iterations = 20
seed = 123
number_of_samples = 10

fun = f3dasm.functions.Schwefel(
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
optimizer.init_parameters()
optimizer.parameter.noise_fix = False
# optimizer.parameter.kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.CosineKernel())
# optimizer.parameter.acquisition = botorch.acquisition.ExpectedImprovement
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

print(res)

# plot_x = np.linspace(fun.input_domain[0, 0], fun.input_domain[0, 1], 500)[:, None]
# plt.plot(plot_x, fun(plot_x))
# plt.scatter(res.data['input'], res.data['output'])
# plt.show()
