import pytest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import f3dasm
from f3dasm.functions import FUNCTIONS, get_functions, AugmentedFunction, MultiFidelityFunction
from f3dasm.design import ExperimentData

import gpytorch
import torch
import math
from sklearn.preprocessing import StandardScaler

from typing import Any

pytestmark = pytest.mark.leo

def test_gpr_gpytorch():

    ###
    dim = 1
    seed = 123
    noisy_data_bool = 1
    numsamples = 100
    fun_class = f3dasm.functions.Sphere
    opt_algo = torch.optim.Adam
    opt_algo_kwargs = dict(lr=0.25)
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

    param = f3dasm.machinelearning.gpr.Sogpr_Parameters(
        kernel=kernel,
        mean=mean,
        noise_fix=1- noisy_data_bool,
        opt_algo=opt_algo,
        opt_algo_kwargs=opt_algo_kwargs,
        verbose_training=False,
        training_iter=training_iter,
        )

    regressor = f3dasm.machinelearning.gpr.Sogpr(
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
    opt_algo = torch.optim.Adam
    opt_algo_kwargs = dict(lr=0.25)
    fun_class = f3dasm.functions.AlpineN2
    training_iter = 50
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel()) \
        #+ gpytorch.kernels.ScaleKernel(gpytorch.kernels.CosineKernel())
    n_test = 500

    reg_parameters = f3dasm.machinelearning.gpr.Sogpr_Parameters(
        likelihood=gpytorch.likelihoods.GaussianLikelihood(),
        kernel=kernel,
        opt_algo=opt_algo,
        opt_algo_kwargs=opt_algo_kwargs,
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
        data=ExperimentData(design=parameter_DesignSpace),
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


def test_cokgj_forrester_gpytorch():
    ###

    dim = 1
    seed = 123
    noisy_data_bool = 1
    # fun_class = f3dasm.functions.Sphere
    opt_algo = torch.optim.Adam
    opt_algo_kwargs = dict(lr=0.5)
    training_iter = 150
    n_test = 500
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    train_data_supplied = 1

    mean_module_list = torch.nn.ModuleList([
        gpytorch.means.ZeroMean(),
        gpytorch.means.ZeroMean()
    ])

    covar_module_list = torch.nn.ModuleList([
        gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
        gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
    ])

    class Forrester(f3dasm.functions.Function):
        def __init__(self, seed: Any or int = None):
            super().__init__(seed)

        def evaluate(self, x):
            return (6 * x - 2) ** 2 * np.sin(12 * x - 4)

    class Forrester_lf(f3dasm.functions.Function):
        def __init__(self, seed: Any or int = None):
            super().__init__(seed)

        def evaluate(self, x):
            return 0.5 * (6 * x - 2) ** 2 * np.sin(12 * x - 4) + 10 * (x - 0.5) - 5

    # base_fun = fun_class(
    #     dimensionality=dim,
    #     scale_bounds=np.tile([0.0, 1.0], (dim, 1)),
    #     )

    fids = [0.5, 1.0]
    costs = [0.5, 1.0]
    samp_nos = [11, 4]

    funs = []
    mf_design_space = []
    mf_sampler = []
    mf_train_data = []

    for fid_no, (fid, cost, samp_no) in enumerate(zip(fids, costs, samp_nos)):

        if fid_no == 0:
            fun = Forrester_lf()
        else:
            fun = Forrester()

        # fun = AugmentedFunction(
        #         base_fun=base_fun,
        #         fid=fid,
        #         )
        
        parameter_DesignSpace = f3dasm.make_nd_continuous_design(
            bounds=np.tile([0.0, 1.0], (dim, 1)),
            dimensionality=dim,
        )

        sampler = f3dasm.sampling.SobolSequence(design=parameter_DesignSpace, seed=seed)

        if train_data_supplied:
            train_data = ExperimentData(design=parameter_DesignSpace)

            if fid_no == 1:
                input_array = np.array([0., 0.4, 0.6, 1.])[:, None]
            else:
                input_array = np.linspace(0, 1, 11)[:, None]

            train_data.add_numpy_arrays(
                input_rows=input_array, 
                output_rows=np.full_like(input_array, np.nan)
                )
        else:
            train_data = sampler.get_samples(numsamples=samp_no)

        output = fun(train_data) 

        train_x = torch.tensor(train_data.get_input_data().values)
        train_y = torch.tensor(train_data.get_output_data().values.flatten())

        # Scaling the output
        if fid_no == 0:
            scaler = StandardScaler()
            scaler.fit(train_y.numpy()[:, None])
        train_y_scaled = torch.tensor(scaler.transform(train_y.numpy()[:, None]).flatten())

        # if fid_no == 0:
        #     train_y_scaled += noisy_data_bool * np.random.randn(*train_y_scaled.shape) * math.sqrt(0.04)

        train_data.add_output(output=train_y_scaled)

        funs.append(fun)
        mf_design_space.append(parameter_DesignSpace)
        mf_sampler.append(sampler)
        mf_train_data.append(train_data)

    ###

    param = f3dasm.machinelearning.gpr.Cokgj_Parameters(
        likelihood=likelihood,
        kernel=covar_module_list,
        mean=mean_module_list,
        noise_fix=1- noisy_data_bool,
        opt_algo=opt_algo,
        opt_algo_kwargs=opt_algo_kwargs,
        verbose_training=0,
        training_iter=training_iter,
        )

    regressor = f3dasm.machinelearning.gpr.Cokgj(
        train_data=mf_train_data, 
        design=train_data.design,
        parameter=param,
    )

    surrogate = regressor.train()

    ###

    # Get into evaluation (predictive posterior) mode
    surrogate.model.eval()
    surrogate.model.likelihood.eval()

    # # Test points are regularly spaced along [0,1]
    # # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_sampler = f3dasm.sampling.SobolSequence(design=parameter_DesignSpace, seed=0)
        
        if dim == 1:
            test_x = torch.linspace(0, 1, n_test)[:, None]
        else:
            test_x = torch.tensor(test_sampler.get_samples(numsamples=n_test).get_input_data().values)
    
        observed_pred = surrogate.predict([torch.tensor([])[:, None], test_x])
        exact_y = fun(test_x.numpy())#[:, None])

    ###

    metrics_df = surrogate.gp_metrics(
        scaler=scaler,
        observed_pred=observed_pred,
        exact_y=exact_y.flatten(),
    )


def test_cokgj_gpytorch():
    ###

    dim = 1
    seed = 123
    noisy_data_bool = 1
    fun_class = f3dasm.functions.Sphere
    opt_algo = torch.optim.Adam
    opt_algo_kwargs = dict(lr=0.5)
    training_iter = 150
    n_test = 500
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    train_data_supplied = 1

    mean_module_list = torch.nn.ModuleList([
        gpytorch.means.ZeroMean(),
        gpytorch.means.ZeroMean()
    ])

    covar_module_list = torch.nn.ModuleList([
        gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
        gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
    ])

    base_fun = fun_class(
        dimensionality=dim,
        scale_bounds=np.tile([0.0, 1.0], (dim, 1)),
        )

    fids = [0.5, 1.0]
    costs = [0.5, 1.0]
    samp_nos = [11, 4]

    funs = []
    mf_design_space = []
    mf_sampler = []
    mf_train_data = []

    for fid_no, (fid, cost, samp_no) in enumerate(zip(fids, costs, samp_nos)):

        fun = AugmentedFunction(
                base_fun=base_fun,
                fid=fid,
                )
        
        parameter_DesignSpace = f3dasm.make_nd_continuous_design(
            bounds=np.tile([0.0, 1.0], (dim, 1)),
            dimensionality=dim,
        )

        sampler = f3dasm.sampling.SobolSequence(design=parameter_DesignSpace, seed=seed)

        if train_data_supplied:
            train_data = ExperimentData(design=parameter_DesignSpace)

            if fid_no == 1:
                input_array = np.array([0., 0.4, 0.6, 1.])[:, None]
            else:
                input_array = np.linspace(0, 1, 11)[:, None]

            train_data.add_numpy_arrays(
                input_rows=input_array, 
                output_rows=np.full_like(input_array, np.nan)
                )
        else:
            train_data = sampler.get_samples(numsamples=samp_no)

        output = fun(train_data) 

        train_x = torch.tensor(train_data.get_input_data().values)
        train_y = torch.tensor(train_data.get_output_data().values.flatten())

        # Scaling the output
        if fid_no == 0:
            scaler = StandardScaler()
            scaler.fit(train_y.numpy()[:, None])
        train_y_scaled = torch.tensor(scaler.transform(train_y.numpy()[:, None]).flatten())

        # if fid_no == 0:
        #     train_y_scaled += noisy_data_bool * np.random.randn(*train_y_scaled.shape) * math.sqrt(0.04)

        train_data.add_output(output=train_y_scaled)

        funs.append(fun)
        mf_design_space.append(parameter_DesignSpace)
        mf_sampler.append(sampler)
        mf_train_data.append(train_data)

    ###

    param = f3dasm.machinelearning.gpr.Cokgj_Parameters(
        likelihood=likelihood,
        kernel=covar_module_list,
        mean=mean_module_list,
        noise_fix=1- noisy_data_bool,
        opt_algo=opt_algo,
        opt_algo_kwargs=opt_algo_kwargs,
        verbose_training=0,
        training_iter=training_iter,
        )

    regressor = f3dasm.machinelearning.gpr.Cokgj(
        train_data=mf_train_data, 
        design=train_data.design,
        parameter=param,
    )

    surrogate = regressor.train()

    ###

    # Get into evaluation (predictive posterior) mode
    surrogate.model.eval()
    surrogate.model.likelihood.eval()

    # # Test points are regularly spaced along [0,1]
    # # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_sampler = f3dasm.sampling.SobolSequence(design=parameter_DesignSpace, seed=0)
        
        if dim == 1:
            test_x = torch.linspace(0, 1, n_test)[:, None]
        else:
            test_x = torch.tensor(test_sampler.get_samples(numsamples=n_test).get_input_data().values)
    
        observed_pred = surrogate.predict([torch.tensor([])[:, None], test_x])
        exact_y = fun(test_x.numpy())#[:, None])

    ###

    metrics_df = surrogate.gp_metrics(
        scaler=scaler,
        observed_pred=observed_pred,
        exact_y=exact_y.flatten(),
    )


def test_mtask_forrester_gpytorch():
    ###

    dim = 1
    seed = 123
    noisy_data_bool = 1
    # fun_class = f3dasm.functions.Sphere
    opt_algo = torch.optim.Adam
    opt_algo_kwargs = dict(lr=0.1)
    training_iter = 150
    n_test = 500
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    train_data_supplied = 1

    mean_module = gpytorch.means.ZeroMean()
    covar_module = gpytorch.kernels.RBFKernel()

    ###

    class Forrester(f3dasm.functions.Function):
        def __init__(self, seed: Any or int = None):
            super().__init__(seed)

        def evaluate(self, x):
            return (6 * x - 2) ** 2 * np.sin(12 * x - 4)

    class Forrester_lf(f3dasm.functions.Function):
        def __init__(self, seed: Any or int = None):
            super().__init__(seed)

        def evaluate(self, x):
            return 0.5 * (6 * x - 2) ** 2 * np.sin(12 * x - 4) + 10 * (x - 0.5) - 5

    # base_fun = fun_class(
    #     dimensionality=dim,
    #     scale_bounds=np.tile([0.0, 1.0], (dim, 1)),
    #     )

    fids = [0.5, 1.0]
    costs = [0.5, 1.0]
    samp_nos = [11, 100]

    funs = []
    mf_design_space = []
    mf_sampler = []
    mf_train_data = []

    for fid_no, (fid, cost, samp_no) in enumerate(zip(fids, costs, samp_nos)):

        # fun = AugmentedFunction(
        #         base_fun=base_fun,
        #         fid=fid,
        #         )

        if fid_no == 0:
            fun = Forrester_lf()
        else:
            fun = Forrester()
        
        parameter_DesignSpace = f3dasm.make_nd_continuous_design(
            bounds=np.tile([0.0, 1.0], (dim, 1)),
            dimensionality=dim,
        )

        sampler = f3dasm.sampling.SobolSequence(design=parameter_DesignSpace, seed=seed)

        if train_data_supplied:
            train_data = ExperimentData(design=parameter_DesignSpace)

            if fid_no == 0:
                input_array = np.linspace(0, 1, 11)[:, None]
            else:
                input_array = np.array([0., 0.4, 0.6, 1.])[:, None]

            train_data.add_numpy_arrays(
                input_rows=input_array, 
                output_rows=np.full_like(input_array, np.nan)
                )
        else:
            train_data = sampler.get_samples(numsamples=samp_no)

        output = fun(train_data) 

        train_x = torch.tensor(train_data.get_input_data().values)
        train_y = torch.tensor(train_data.get_output_data().values.flatten())

        # Scaling the output
        if fid_no == 0:
            scaler = StandardScaler()
            scaler.fit(train_y.numpy()[:, None])
        train_y_scaled = torch.tensor(scaler.transform(train_y.numpy()[:, None]).flatten())

        # if fid_no == 0:
        #     train_y_scaled += noisy_data_bool * np.random.randn(*train_y_scaled.shape) * math.sqrt(0.04)

        train_data.add_output(output=train_y_scaled)

        funs.append(fun)
        mf_design_space.append(parameter_DesignSpace)
        mf_sampler.append(sampler)
        mf_train_data.append(train_data)

    ###

    param = f3dasm.machinelearning.gpr.Multitask_Parameters(
        likelihood=likelihood,
        kernel=covar_module,
        mean=mean_module,
        noise_fix=1- noisy_data_bool,
        opt_algo=opt_algo,
        opt_algo_kwargs=opt_algo_kwargs,
        verbose_training=0,
        training_iter=training_iter,
        )

    regressor = f3dasm.machinelearning.gpr.MultitaskGPR(
        mf_train_data=mf_train_data, 
        design=train_data.design,
        parameter=param,
    )

    surrogate = regressor.train()

    ###

    # Get into evaluation (predictive posterior) mode
    surrogate.model.eval()
    surrogate.model.likelihood.eval()

    # # Test points are regularly spaced along [0,1]
    # # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_sampler = f3dasm.sampling.SobolSequence(design=parameter_DesignSpace, seed=0)
        
        if dim == 1:
            test_x = torch.linspace(0, 1, n_test)[:, None]
        else:
            test_x = torch.tensor(test_sampler.get_samples(numsamples=n_test).get_input_data().values)
    
        observed_pred = surrogate.predict([torch.tensor([])[:, None], test_x])
        # observed_pred = surrogate.predict([test_x, torch.tensor([])[:, None]])
        # observed_pred = surrogate.predict(test_x)
        exact_y = fun(test_x.numpy())#[:, None])

    ###

    if dim == 1:
        surrogate.plot_gpr(
            test_x=test_x.flatten(), 
            scaler=scaler, 
            exact_y=exact_y, 
            observed_pred=observed_pred,
            train_x=train_x,
            train_y=torch.tensor(scaler.inverse_transform(train_y_scaled.numpy()[:, None]))
            # train_x=torch.tensor(mf_train_data[0].data.input.x0.values),
            # train_y=torch.tensor(scaler.inverse_transform(mf_train_data[0].data.output.y.values[:, None])),
            )

    # surrogate.plot_mll(
    #     train_x=train_x,
    #     train_y_scaled=train_y_scaled,
    # )

    metrics_df = surrogate.gp_metrics(
        scaler=scaler,
        observed_pred=observed_pred,
        exact_y=exact_y.flatten(),
    )


def test_mfbo_cokgj_gpytorch():
    dim = 1
    seed = 123
    noisy_data_bool = 1
    fun_class = f3dasm.functions.AlpineN2
    training_iter = 50
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    opt_algo = torch.optim.Adam
    opt_algo_kwargs = dict(lr=0.1)

    base_fun = fun_class(
        dimensionality=dim,
        scale_bounds=np.tile([0.0, 1.0], (dim, 1)),
        )

    mean_module_list = torch.nn.ModuleList([
        gpytorch.means.ZeroMean(),
        gpytorch.means.ZeroMean()
    ])

    covar_module_list = torch.nn.ModuleList([
        gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
        gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel()),
    ])

    reg_parameters = f3dasm.machinelearning.gpr.Cokgj_Parameters(
        likelihood=likelihood,
        kernel=covar_module_list,
        mean=mean_module_list,
        noise_fix=1- noisy_data_bool,
        opt_algo=opt_algo,
        opt_algo_kwargs=opt_algo_kwargs,
        verbose_training=0,
        training_iter=training_iter,
        )

    ### opt parameters

    iterations = 1
    numbers_of_samples = [20, 5]
    fidelity_parameters = [0.5, 1.]
    costs = [0.5, 1.]
    budget = 10

    ###

    fidelity_functions = []
    multifidelity_samplers = []

    for i, fidelity_parameter in enumerate(fidelity_parameters):

        fun = AugmentedFunction(
                base_fun=base_fun,
                fid=fidelity_parameter,
                dimensionality=base_fun.dimensionality,
                scale_bounds=base_fun.scale_bounds,
                )
        
        parameter_DesignSpace = f3dasm.make_nd_continuous_design(
            bounds=np.tile([0.0, 1.0], (dim, 1)),
            dimensionality=dim,
        )
        # fidelity_parameter = f3dasm.ConstantParameter(name="fid", constant_value=fidelity_parameter)
        # parameter_DesignSpace.add_input_space(fidelity_parameter)

        sampler = f3dasm.sampling.SobolSequence(design=parameter_DesignSpace, seed=seed)

        fidelity_functions.append(fun)
        multifidelity_samplers.append(sampler)

    multifidelity_function = MultiFidelityFunction(
        fidelity_functions=fidelity_functions,
        fidelity_parameters=fidelity_parameters,
        costs=costs,
    )

    optimizer = f3dasm.optimization.MFBayesianOptimizationTorch(
        data=ExperimentData(design=parameter_DesignSpace),
        multifidelity_function=multifidelity_function,
    )

    # optimizer.init_parameters()
    optimizer.parameter.n_init = numbers_of_samples
    optimizer.parameter.regressor_hyperparameters = reg_parameters
    # optimizer.parameter.regressor_hyperparameters.noise_fix = True
    # optimizer.parameter.kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.CosineKernel())
    # optimizer.parameter.acquisition = f3dasm.base.acquisition.ExpectedImprovement
    # optimizer.parameter.acquisition_hyperparameters = f3dasm.optimization.bayesianoptimization_torch.Acquisition_Parameters(
    #     best_f=-np.inf,
    #     maximize=False,
    # )
    optimizer.parameter.visualize_gp = True
    # optimizer.parameter.acquisition_hyperparameters = {
    #     'best_f': np.inf,
    #     'maximize': False,
    # }

    res = f3dasm.run_multi_fidelity_optimization(
        optimizer=optimizer,
        multifidelity_function=multifidelity_function,
        multifidelity_samplers=multifidelity_samplers,
        iterations=iterations,
        seed=seed,
        numbers_of_samples=numbers_of_samples,
        budget=budget,
    )

if __name__ == "__main__":  # pragma: no cover
    pytest.main()