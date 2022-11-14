import numpy as np
from matplotlib import pyplot as plt

import f3dasm

dim = 1
iterations = 40
seed = 123
number_of_samples = 10
samp_no = 8

fun = f3dasm.functions.Schwefel(
    dimensionality=dim,
    scale_bounds=np.tile([0.0, 1.0], (dim, 1)),
    )

parameter_DesignSpace = f3dasm.make_nd_continuous_design(
    bounds=fun.input_domain.astype(float),
    dimensionality=dim,
)

sampler = f3dasm.sampling.SobolSequenceSampling(design=parameter_DesignSpace)

init_train_data = sampler.get_samples(numsamples=samp_no)
init_train_data.add_output(output=fun(init_train_data))

optimizer = f3dasm.optimization.BayesianOptimizationTorch(
    # data=f3dasm.Data(design=parameter_DesignSpace),
    data=init_train_data,
    )
optimizer.init_parameters()

res = f3dasm.run_optimization(
    optimizer=optimizer,
    function=fun,
    sampler=sampler,
    iterations=iterations,
    seed=seed,
    number_of_samples=number_of_samples,
)

plot_x = np.linspace(fun.input_domain[0, 0], fun.input_domain[0, 1], 500)[:, None]
plt.plot(plot_x, fun(plot_x))
plt.scatter(res.data['input'], res.data['output'])
plt.show()
