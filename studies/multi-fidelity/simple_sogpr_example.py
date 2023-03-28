import matplotlib.pyplot as plt
import numpy as np

import f3dasm

dim = 1

fun_class = f3dasm.functions.Sphere

fun = fun_class(
    dimensionality=dim,
    scale_bounds=np.tile([0.0, 1.0], (dim, 1)),
    )

parameter_DesignSpace: f3dasm.DesignSpace = f3dasm.make_nd_continuous_design(
    bounds=np.tile([0.0, 1.0], (dim, 1)),
    dimensionality=dim,
)

sampler = f3dasm.sampling.SobolSequence(design=parameter_DesignSpace)

train_data: f3dasm.Data = sampler.get_samples(numsamples=10)
opt_retries = 50

train_data.add_output(output=fun(train_data))

regressor = f3dasm.regression.gpr.Sogpr(
    train_data=train_data, 
    design=train_data.design,
    noise_fix=True,
)

surrogate = regressor.train(optimize=True, max_retries=opt_retries)

# print(surrogate.model.covar_module.raw_outputscale)
# print(surrogate.model.covar_module.base_kernel.raw_lengthscale)

x_plot = np.linspace(0, 1, 500)[:, None] # add 1D plot method for functions!
y_plot = fun(x_plot)

x_plot_data = f3dasm.Data(design=train_data.design)
x_plot_data.add_numpy_arrays(input=x_plot, output=x_plot)
pred_mean, pred_var = surrogate.predict(test_input_data=x_plot_data)

ucb, lcb = [pred_mean + 2 * (-1) ** k * np.sqrt(np.abs(pred_var)) for k in range(2)]

plt.figure('botorch')
plt.plot(x_plot, y_plot, 'b--', label='Exact') # add regression plot
plt.scatter(train_data.data['input'], train_data.data['output'], c='b', label='Training data')
plt.plot(x_plot, pred_mean, color='purple', label='Prediction')
plt.fill_between(x_plot.flatten(), lcb.flatten(), ucb.flatten(), color='purple', alpha=.25, label='Confidence')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.tight_layout()

###

import sklearn.gaussian_process

kernel_sklearn = sklearn.gaussian_process.kernels.ConstantKernel() * sklearn.gaussian_process.kernels.RBF()
regressor_sklearn = sklearn.gaussian_process.GaussianProcessRegressor(
    kernel=kernel_sklearn, 
    n_restarts_optimizer=opt_retries,
    )
surrogate_sklearn = regressor_sklearn.fit(
    X=train_data.get_input_data(), 
    y=train_data.get_output_data()
    )
mean_sklearn, std_sklearn = surrogate_sklearn.predict(X=x_plot, return_std=True)

ucb_sklearn, lcb_sklearn = [mean_sklearn + 2 * (-1) ** k * std_sklearn for k in range(2)]

plt.figure('sklearn')
plt.plot(x_plot, y_plot, 'b--', label='Exact') # add regression plot
plt.scatter(train_data.data['input'], train_data.data['output'], c='b', label='Training data')
plt.plot(x_plot, mean_sklearn, color='purple', label='Prediction')
plt.fill_between(x_plot.flatten(), lcb_sklearn.flatten(), ucb_sklearn.flatten(), color='purple', alpha=.25, label='Confidence')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.tight_layout()

###

import GPy

kernel_gpy = GPy.kern.RBF(input_dim=1)
regressor_gpy = GPy.models.GPRegression(
    X=train_data.get_input_data(), 
    Y=train_data.get_output_data(), 
    kernel=kernel_gpy
    )
regressor_gpy.Gaussian_noise.variance.fix(0)

# regressor_gpy.optimize()
regressor_gpy.optimize_restarts(num_restarts=opt_retries, verbose=False)
surrogate_gpy = regressor_gpy

mean_gpy, var_gpy = surrogate_gpy.predict(Xnew=x_plot)

ucb_gpy, lcb_gpy = [mean_gpy + 2 * (-1) ** k * np.sqrt(np.abs(var_gpy)) for k in range(2)]

plt.figure('GPy')
plt.plot(x_plot, y_plot, 'b--', label='Exact') # add regression plot
plt.scatter(train_data.data['input'], train_data.data['output'], c='b', label='Training data')
plt.plot(x_plot, mean_gpy, color='purple', label='Prediction')
plt.fill_between(x_plot.flatten(), lcb_gpy.flatten(), ucb_gpy.flatten(), color='purple', alpha=.25, label='Confidence')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.tight_layout()

###

print()
print('botorch')
print(surrogate.model.covar_module.raw_outputscale)
print(surrogate.model.covar_module.base_kernel.raw_lengthscale)
print()
print('sklearn')
print(surrogate_sklearn.kernel_.get_params())
print()
print('GPy')
print(surrogate_gpy)

plt.show()