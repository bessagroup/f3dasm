import matplotlib.pyplot as plt
import numpy as np

import f3dasm

dim = 1
fun = f3dasm.functions.Schwefel(dimensionality=dim)
parameter_DesignSpace: f3dasm.DesignSpace = f3dasm.make_nd_continuous_design(
    bounds=fun.input_domain.astype(float),
    dimensionality=dim,
)
SobolSampler = f3dasm.sampling.SobolSequenceSampling(design=parameter_DesignSpace)

samples: f3dasm.Data = SobolSampler.get_samples(numsamples=8)
samples.add_output(output=fun(samples))

regressor = f3dasm.regression.gpr.Sogpr(
    train_data=samples, design=parameter_DesignSpace
)
surrogate = regressor.train()

plot_x = np.linspace(fun.input_domain[0, 0], fun.input_domain[0, 1], 500)[:, None]
mean, var = surrogate.predict(test_input_data=plot_x)

ucb, lcb = [mean + (-1) ** k * 2 * np.sqrt(np.abs(var)) for k in range(2)]

plt.plot(plot_x, fun(plot_x))
plt.plot(plot_x, mean, "r--")
plt.plot(plot_x, lcb, "k", linewidth=0.5)
plt.plot(plot_x, ucb, "k", linewidth=0.5)
plt.fill_between(plot_x.flatten(), lcb.flatten(), ucb.flatten(), color="r", alpha=0.1)
plt.scatter(samples.get_input_data(), samples.get_output_data(), c="r")
plt.grid()
plt.tight_layout()

plt.show()
