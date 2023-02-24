import GPy
import matplotlib.pyplot as plt
import numpy as np

import torch
import math
torch.random.manual_seed(0)
# Training data is 100 points in [0, 1] inclusive regularly spaced
train_x = torch.linspace(0, 1, 10)
# True function is sin(2*pi*x) with Gaussian noise
train_y = torch.sin(train_x * (2 * math.pi)) #+ torch.randn(train_x.size()) * math.sqrt(0.04)

X_train = train_x.numpy()[:, None]
y_train = train_y.numpy()[:, None]

# X = np.random.uniform(-3.,3.,(20,1))
# Y = np.sin(X)# + np.random.randn(20,1)*0.05

kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)

m = GPy.models.GPRegression(X_train, y_train, kernel)

from IPython.display import display
display(m)

fig = m.plot()

m.optimize(messages=False)
# m.optimize_restarts(num_restarts = 10)

display(m)
fig = m.plot()

amplitude_scale = np.logspace(np.log10(m.rbf.variance) - 0.1, np.log10(m.rbf.variance) + 0.1, num=30)
length_scale = np.logspace(np.log10(m.rbf.lengthscale) - 0.1, np.log10(m.rbf.lengthscale) + 0.1, num=30)
noise_level = np.logspace(-3, 1, num=50)
# length_scale_grid, noise_scale_grid = np.meshgrid(length_scale, noise_level)
length_scale_grid, amplitude_scale_grid = np.meshgrid(length_scale, amplitude_scale)

log_marginal_likelihood_list = []

for length_scale, amplitude_scale in zip(length_scale_grid.ravel(), amplitude_scale_grid.ravel()):
    m.rbf.lengthscale = length_scale
    m.rbf.variance = amplitude_scale
    log_marginal_likelihood_list.append(m._log_marginal_likelihood)

log_marginal_likelihood = np.array(log_marginal_likelihood_list).reshape(length_scale_grid.shape)

# vmin, vmax = (-log_marginal_likelihood).min(), (-log_marginal_likelihood).max()
# level = np.logspace(np.log10(vmin), np.log10(vmax), num=250)
plt.figure()
plt.contour(
    length_scale_grid,
    # noise_scale_grid,
    amplitude_scale_grid,
    -log_marginal_likelihood,
    levels = 250
    # levels=level,
    # norm=LogNorm(vmin=vmin, vmax=vmax),
)
plt.colorbar()
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Length-scale")
# plt.ylabel("Noise-level")
plt.ylabel("Amplitude")
plt.title("Log-marginal-likelihood")

plt.show()

