import numpy as np

def target_generator(X, add_noise=False):
    # target = 0.5 + np.sin(3 * X)
    target = (10.24 * (X - 0.5)) ** 2#np.sin(2 * np.pi * X)
    if add_noise:
        rng = np.random.RandomState(1)
        # target += rng.normal(0, 0.3, size=target.shape)
        target += rng.normal(0, 0.2, size=target.shape)
    return target.squeeze()

# X = np.linspace(0, 5, num=30).reshape(-1, 1)
X = np.linspace(0, 1, num=500).reshape(-1, 1)
y = target_generator(X, add_noise=False)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(X, y, label="Expected signal")
plt.legend()
plt.xlabel("X")
_ = plt.ylabel("y")

# rng = np.random.RandomState(0)
# # X_train = rng.uniform(0, 5, size=20).reshape(-1, 1)
# X_train = np.linspace(0, 1, num=100).reshape(-1, 1)
# y_train = target_generator(X_train, add_noise=True)
# print(y_train)

import torch
import math
torch.random.manual_seed(0)
# Training data is 100 points in [0, 1] inclusive regularly spaced
train_x = torch.linspace(0, 1, 10)
# True function is sin(2*pi*x) with Gaussian noise
train_y = (10.24 * (train_x - 0.5)) ** 2#torch.sin(train_x * (2 * math.pi))# + torch.randn(train_x.size()) * math.sqrt(0.04)

X_train = train_x.numpy()[:, None]
y_train = train_y.numpy()

plt.figure()
plt.plot(X, y, label="Expected signal")
plt.scatter(
    x=X_train[:, 0],
    y=y_train,
    color="black",
    alpha=0.4,
    label="Observations",
)
plt.legend()
plt.xlabel("X")
_ = plt.ylabel("y")

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# kernel = 1.0 * RBF(length_scale=1e1, length_scale_bounds=(1e-2, 1e3))\
#      + WhiteKernel(noise_level=1, noise_level_bounds=(1e-5, 1e1))
# gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.0)
# gpr.fit(X_train, y_train)
# y_mean, y_std = gpr.predict(X, return_std=True)

# plt.figure()
# plt.plot(X, y, label="Expected signal", linestyle='--')
# plt.scatter(x=X_train[:, 0], y=y_train, color="black", alpha=0.4, label="Observations")
# # plt.errorbar(X, y_mean, y_std)
# p = plt.plot(X, y_mean, label='Mean')
# c = p[-1].get_color()
# plt.fill_between(X.flatten(), y_mean - 2 * y_std, y_mean + 2 * y_std, alpha=0.5, color=c, label='Confidence interval')
# plt.legend()
# plt.xlabel("X")
# plt.ylabel("y")
# _ = plt.title(
#     f"Initial: {kernel}\nOptimum: {gpr.kernel_}\nLog-Marginal-Likelihood: "
#     f"{gpr.log_marginal_likelihood(gpr.kernel_.theta)}",
#     fontsize=8,
# )

kernel = 1.0 * RBF(length_scale=1e-1, length_scale_bounds=(1e-2, 1e3))\
    #  + WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-10, 1e1))
gpr = GaussianProcessRegressor(kernel=kernel)#, alpha=0.0)
gpr.fit(X_train, y_train)
y_mean, y_std = gpr.predict(X, return_std=True)

plt.figure()
plt.plot(X, y, label="Expected signal", linestyle='--')
plt.scatter(x=X_train[:, 0], y=y_train, color="black", alpha=0.4, label="Observations")
# plt.errorbar(X, y_mean, y_std)
p = plt.plot(X, y_mean, label='Mean')
c = p[-1].get_color()
plt.fill_between(X.flatten(), y_mean - 2 * y_std, y_mean + 2 * y_std, alpha=0.5, color=c, label='Confidence interval')
plt.legend()
plt.xlabel("X")
plt.ylabel("y")
_ = plt.title(
    f"Initial: {kernel}\nOptimum: {gpr.kernel_}\nLog-Marginal-Likelihood: "
    f"{gpr.log_marginal_likelihood(gpr.kernel_.theta)}",
    fontsize=8,
)

print(np.exp(gpr.kernel_.theta))

# from matplotlib.colors import LogNorm

opt_amp, opt_scale = np.log10(np.exp(gpr.kernel_.theta))

amplitude_scale = np.logspace(opt_amp - 0.1, opt_amp + 0.1, num=50)
length_scale = np.logspace(opt_scale - 0.1, opt_scale + 0.1, num=50)
noise_level = np.logspace(-3, 1, num=50)
# length_scale_grid, noise_scale_grid = np.meshgrid(length_scale, noise_level)
length_scale_grid, amplitude_scale_grid = np.meshgrid(length_scale, amplitude_scale)

log_marginal_likelihood = [
    # gpr.log_marginal_likelihood(theta=np.log([np.exp(gpr.kernel_.theta)[0], scale, noise]))
    # for scale, noise in zip(length_scale_grid.ravel(), noise_scale_grid.ravel())
    gpr.log_marginal_likelihood(theta=np.log([amp, scale]))
    for scale, amp in zip(length_scale_grid.ravel(), amplitude_scale_grid.ravel())
]
log_marginal_likelihood = np.reshape(
    log_marginal_likelihood, newshape=length_scale_grid.shape
)

# vmin, vmax = (-log_marginal_likelihood).min(), (-log_marginal_likelihood).max()
# level = np.logspace(np.log10(vmin), np.log10(vmax), num=250)
plt.figure()
plt.contour(
    length_scale_grid,
    # noise_scale_grid,
    amplitude_scale_grid,
    -log_marginal_likelihood,
    # levels=level,
    levels=250,
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

