from matplotlib import pyplot as plt

import f3dasm

dim = 1
fun = f3dasm.functions.Schwefel(dimensionality=dim)
parameter_DesignSpace = f3dasm.make_nd_continuous_design(
    bounds=fun.input_domain.astype(float),
    dimensionality=dim,
)
SobolSampler = f3dasm.sampling.SobolSequenceSampling(design=parameter_DesignSpace)
samples = SobolSampler.get_samples(numsamples=8)
samples.add_output(output=fun(samples))
optimizer = f3dasm.optimization.BayesianOptimizationTorch(data=samples)
optimizer.init_parameters()
res = f3dasm.run_optimization(
    optimizer=optimizer,
    function=fun,
    sampler=SobolSampler,
    iterations=40,
    seed=123,
    number_of_samples=10,
)

# plot_x = np.linspace(fun.input_domain[0, 0], fun.input_domain[0, 1], 500)[:, None]
# plt.plot(plot_x, fun(plot_x))
# plt.scatter(res.data['input'], res.data['output'])
# plt.show()
