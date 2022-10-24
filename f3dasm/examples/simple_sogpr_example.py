import matplotlib.pyplot as plt
import numpy as np

from f3dasm import make_nd_continuous_design
from f3dasm.functions import pybenchfunction
from f3dasm.regression.gpr import Sogpr
from f3dasm.sampling import SobolSequenceSampling

dim = 1
fun = pybenchfunction.Exponential(dimensionality=dim)
parameter_DesignSpace = make_nd_continuous_design(
    bounds=fun.input_domain.astype(float),
    dimensionality=dim,
)
SobolSampler = SobolSequenceSampling(design=parameter_DesignSpace)

samples = SobolSampler.get_samples(numsamples=8)
samples.add_output(output=fun(samples))
# print(samples.data['input'])
# print(parameter_DesignSpace.input_space[0].name)

# train_x = SobolSampler.sample_continuous(numsamples=8)
train_x = samples.data['input'].values
# train_y = fun(train_x)
train_y = samples.data['output'].values

regressor = Sogpr(
    train_input_data=train_x,
    train_output_data=train_y,
    design=parameter_DesignSpace
)
surrogate = regressor.train()

plot_x = np.linspace(fun.input_domain[0, 0], fun.input_domain[0, 1], 500)[:, None]
mean, var = surrogate.predict(test_input_data=plot_x)

ucb, lcb = [mean + (-1) ** k * 2 * np.sqrt(np.abs(var)) for k in range(2)]

plt.plot(plot_x, fun(plot_x))
plt.plot(plot_x, mean, 'r--')
plt.plot(plot_x, lcb, 'k', linewidth=.5)
plt.plot(plot_x, ucb, 'k', linewidth=.5)
plt.fill_between(plot_x.flatten(), lcb.flatten(), ucb.flatten(), color='r', alpha=0.1)
plt.scatter(train_x, train_y, c='r')
plt.grid()
plt.tight_layout()

plt.show()
