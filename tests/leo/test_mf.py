import pytest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import f3dasm
from f3dasm.base.function import AugmentedFunction, MultiFidelityFunction
from f3dasm.functions import FUNCTIONS, get_functions

import gpytorch
import torch
import math
from sklearn.preprocessing import StandardScaler

pytestmark = pytest.mark.leo

def test_gpr_gpytorch():

    ###
    dim = 1
    seed = 123
    noisy_data_bool = 1
    numsamples = 100
    fun_class = f3dasm.functions.Sphere
    opt_algo = torch.optim.Adam
    training_iter = 50
    mean = gpytorch.means.ZeroMean()
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel()) \
        #+ gpytorch.kernels.ScaleKernel(gpytorch.kernels.CosineKernel())
    # plot_mll = 1
    # plot_gpr = 1
    # train_surrogate = True
    n_test = 500

    ###

    fun = fun_class(
        dimensionality=dim,
        scale_bounds=np.tile([0.0, 1.0], (dim, 1)),
        )

    parameter_DesignSpace: f3dasm.DesignSpace = f3dasm.make_nd_continuous_design(
        bounds=np.tile([0.0, 1.0], (dim, 1)),
        dimensionality=dim,
    )

    sampler = f3dasm.sampling.SobolSequence(design=parameter_DesignSpace, seed=seed)

    train_data: f3dasm.Data = sampler.get_samples(numsamples=numsamples)
    # opt_retries = 50

    output = fun(train_data) 
    # train_data.add_output(output=output)

    train_x = torch.tensor(train_data.get_input_data().values)
    train_y = torch.tensor(train_data.get_output_data().values.flatten())

    scaler = StandardScaler()
    scaler.fit(train_y.numpy()[:, None])
    train_y_scaled = torch.tensor(scaler.transform(train_y.numpy()[:, None]).flatten())

    train_y_scaled += noisy_data_bool * np.random.randn(*train_y_scaled.shape) * math.sqrt(0.04)

    train_data.add_output(output=train_y_scaled)

    ###

    param = f3dasm.regression.gpr.Sogpr_Parameters(
        kernel=kernel,
        mean=mean,
        noise_fix=1- noisy_data_bool,
        opt_algo=opt_algo,
        verbose_training=False,
        training_iter=training_iter,
        )

    regressor = f3dasm.regression.gpr.Sogpr(
        train_data=train_data, 
        design=train_data.design,
        parameter=param,
        # noise_fix=1 - noisy_data_bool,
    )

    surrogate = regressor.train(
        # opt_algo=opt_algo,
        # verbose_optimization=False,
        # training_iter=training_iter,
    )

    ###

    # Get into evaluation (predictive posterior) mode
    surrogate.model.eval()
    surrogate.model.likelihood.eval()

    # # Test points are regularly spaced along [0,1]
    # # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_sampler = f3dasm.sampling.SobolSequence(design=parameter_DesignSpace, seed=0)
        
        if dim == 1:
            test_x = torch.linspace(0, 1, n_test)
        else:
            test_x = torch.tensor(test_sampler.get_samples(numsamples=n_test).get_input_data().values)
        
        # observed_pred = surrogate.model.likelihood(surrogate.model(test_x))
        observed_pred = surrogate.predict(test_x)
        exact_y = fun(test_x.numpy()[:, None])

    ###

    if dim == 1:
        surrogate.plot_gpr(
            test_x=test_x, 
            scaler=scaler, 
            exact_y=exact_y, 
            observed_pred=observed_pred,
            train_x=train_x,
            train_y=torch.tensor(scaler.inverse_transform(train_y_scaled.numpy()[:, None]))
            )

    surrogate.plot_mll(
        train_x=train_x,
        train_y_scaled=train_y_scaled,
    )


def test_bo_gpytorch():
    ### reg parameters

    dim = 1
    seed = None
    noisy_data_bool = 0
    numsamples = 100
    fun_class = f3dasm.functions.AlpineN2
    training_iter = 50
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel()) \
        #+ gpytorch.kernels.ScaleKernel(gpytorch.kernels.CosineKernel())
    n_test = 500

    reg_parameters = f3dasm.regression.gpr.Sogpr_Parameters(
        likelihood=gpytorch.likelihoods.GaussianLikelihood(),
        kernel=kernel,
        noise_fix=1 - noisy_data_bool,
        training_iter=training_iter,
    )

    ###

    ### opt parameters

    iterations = 10
    number_of_samples = 5

    ###

    fun = fun_class(
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

    optimizer.parameter.n_init = number_of_samples
    optimizer.parameter.regressor_hyperparameters = reg_parameters
    optimizer.parameter.acquisition = f3dasm.base.acquisition.ExpectedImprovement
    optimizer.parameter.acquisition_hyperparameters = f3dasm.optimization.bayesianoptimization_torch.Acquisition_Parameters(
        best_f=-np.inf,
        maximize=False,
    )
    optimizer.parameter.visualize_gp = True

    res = f3dasm.run_optimization(
        optimizer=optimizer,
        function=fun,
        sampler=sampler,
        iterations=iterations,
        seed=seed,
        number_of_samples=number_of_samples,
    )

    res.data['output'] = optimizer.scaler.inverse_transform(res.data['output'])


if __name__ == "__main__":  # pragma: no cover
    pytest.main()