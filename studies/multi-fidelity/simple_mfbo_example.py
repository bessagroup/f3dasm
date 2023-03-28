import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import f3dasm
from f3dasm import AugmentedFunction, Data
from f3dasm.base import function

dim = 1
iterations = 10
seed = 123
numbers_of_samples = [20, 5]

fidelity_parameters = [0.5, 1.]
costs = [0.5, 1.]

base_fun = f3dasm.functions.Schwefel(
    dimensionality=dim,
    scale_bounds=np.tile([0.0, 1.0], (dim, 1)),
    )

fidelity_functions = []
multifidelity_samplers = []

for i, fidelity_parameter in enumerate(fidelity_parameters):

    fun = AugmentedFunction(
            base_fun=base_fun,
            fid=fidelity_parameter,
            )
    
    parameter_DesignSpace = f3dasm.make_nd_continuous_design(
        bounds=np.tile([0.0, 1.0], (dim, 1)),
        dimensionality=dim,
    )
    fidelity_parameter = f3dasm.ConstantParameter(name="fid", constant_value=fidelity_parameter)
    parameter_DesignSpace.add_input_space(fidelity_parameter)

    sampler = f3dasm.sampling.SobolSequence(design=parameter_DesignSpace)

    fidelity_functions.append(fun)
    multifidelity_samplers.append(sampler)

multifidelity_function = function.MultiFidelityFunction(
    fidelity_functions=fidelity_functions,
    fidelity_parameters=fidelity_parameters,
    costs=costs,
)

optimizer = f3dasm.optimization.MFBayesianOptimizationTorch(
    data=f3dasm.Data(design=parameter_DesignSpace),
    multifidelity_function=multifidelity_function,
)

optimizer.init_parameters()

res = f3dasm.run_multi_fidelity_optimization(
    optimizer=optimizer,
    multifidelity_function=multifidelity_function,
    multifidelity_samplers=multifidelity_samplers,
    iterations=iterations,
    seed=123,
    numbers_of_samples=numbers_of_samples,
    budget=10 # add budget to optimization iterator
)

print(res[-1].data)

# res[-1].data.to_csv('test_hist.csv')

# x_plot = np.linspace(0, 1, 500)[:, None]
# y_plot_high = base_fun(x_plot)
# plt.plot(x_plot, y_plot_high)

# init_data = res[-1].data.loc[:np.sum(samp_nos)]
# init_data_x = init_data['input', 'x0']
# init_data_y = init_data['output']
# plt.scatter(init_data_x, init_data_y, color='black')#, alpha=np.arange(len(init_data_y) + 1) / (len(init_data_y) + 1))
# plt.show()

# plot_x = np.linspace(fun.base_fun.input_domain[0, 0], fun.base_fun.input_domain[0, 1], 500)[:, None]
# plt.plot(plot_x, fun(np.hstack((np.ones_like(plot_x), plot_x))))
# plt.scatter(res[1].data['input', 'x0'], res[1].data['output'])
# plt.show()