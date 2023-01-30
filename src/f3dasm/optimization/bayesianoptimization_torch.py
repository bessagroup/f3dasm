from dataclasses import dataclass, field
from typing import Any, Tuple, List

import scipy
import numpy as np
import pandas as pd
import torch
from botorch.acquisition import InverseCostWeightedUtility#, UpperConfidenceBound
from botorch.models import AffineFidelityCostModel
from botorch.optim import optimize_acqf, optimize_acqf_mixed

from .. import Optimizer, Function, OptimizerParameters, Data, MultiFidelityFunction
from ..base.acquisition import VFUpperConfidenceBound, UpperConfidenceBound
from ..regression.gpr import Cokgj, Sogpr, Stmf, Sogpr_Parameters
import f3dasm
from sklearn.preprocessing import StandardScaler

import gpytorch

@dataclass
class Acquisition_Parameters():
    """Hyperparameters for Acquisition function"""
    
    best_f: Any = None
    beta: float = 0.4
    maximize: bool = False


@dataclass
class BayesianOptimizationTorch_Parameters(OptimizerParameters):
    """Hyperparameters for BayesianOptimizationTorch optimizer"""

    regressor: f3dasm.Regressor = Sogpr
    acquisition: Any = UpperConfidenceBound
    regressor_hyperparameters: Sogpr_Parameters = Sogpr_Parameters()
    acquisition_hyperparameters: Acquisition_Parameters = Acquisition_Parameters()
    n_init: int = 10
    visualize_gp: bool = False


def cost_model(lf, cost_ratio):
    a = (1 - 1 / cost_ratio) / (1 - lf)
    cost_model = AffineFidelityCostModel(
        fidelity_weights={-1: a},
        fixed_cost=1 - a
    )
    return cost_model


def cost_aware_utility(self):
    return InverseCostWeightedUtility(cost_model=self.cost_model())


def optimize_acq_and_get_observation(acq_f, function: Function) -> Tuple[Any, Any]:
    candidates, _ = optimize_acqf(
        acq_function=acq_f,
        # bounds=torch.tensor(function.input_domain.T, dtype=torch.float64),
        bounds=torch.tile(torch.tensor([[0.0], [1.0]]), (1, function.dimensionality)),
        q=1,
        num_restarts=10,
        raw_samples=15,
    )
    new_x = candidates.numpy()
    new_obj = function(new_x)
    return new_x, new_obj


def optimize_mfacq_and_get_observation(acq_f, cost_model, lf, multifidelity_function: MultiFidelityFunction) -> Tuple[Any, Any, Any]:

    dim = multifidelity_function.fidelity_functions[0].dimensionality
    fixed_features_list = [{dim: float(lf)},
                           {dim: 1.0}]

    # design_bounds = torch.tensor(base_fun.input_domain.T, dtype=torch.float64)
    design_bounds = torch.tile(torch.tensor([[0.0], [1.0]]), (1, dim))
    fidelity_bounds = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    augmented_bounds = torch.hstack((design_bounds, fidelity_bounds))

    candidates, _ = optimize_acqf_mixed(
        acq_function=acq_f,
        bounds=augmented_bounds,
        fixed_features_list=fixed_features_list,
        q=1,
        num_restarts=10,
        raw_samples=15,
        options={"batch_limit": 1, "maxiter": 200},
        sequential=True,
    )

    cost = cost_model(lf, cost_ratio=2)(candidates).sum()
    new_x = candidates.numpy()
    function = multifidelity_function.get_fidelity_function_by_parameter(fidelity_parameter=new_x[0, -1])
    new_obj = function(new_x)

    print(new_x, new_obj)

    return new_x, new_obj, cost


def optimize_acquisition(acq_f, function):
    res = scipy.optimize.minimize(
        fun=lambda x: -acq_f(x), 
        x0=np.array([0.5] * function.dimensionality),
        method='L-BFGS-B', 
        bounds=scipy.optimize.Bounds(lb=function.scale_bounds[:, 0], ub=function.scale_bounds[:, 1])
     )
    new_x = res.x[None, :]
    new_obj = function(new_x)
    return new_x, new_obj

class BayesianOptimizationTorch(Optimizer):
    """Bayesian optimization implementation from the botorch library"""
    """Bayesian optimization implementation based on the gpytorch library"""

    parameter: BayesianOptimizationTorch_Parameters = BayesianOptimizationTorch_Parameters()

    # def init_parameters(self):

    #     # Default acquisition hyperparameters
    #     options = {
    #         "acquisition_hyperparameters": {
    #             'beta': 4,
    #             'maximize': False,
    #         }
    #     }

    #     self.parameter = BayesianOptimizationTorch_Parameters(**options)

    def set_algorithm(self):
        self.algorithm = None

    def scale_output(self):
        scaler = StandardScaler()
        self.data.data['output'] = scaler.fit_transform(self.data.data['output'])
        return scaler

    def plot_gpr(self, surrogate, function):
        # Get into evaluation (predictive posterior) mode
        surrogate.model.eval()
        surrogate.model.likelihood.eval()

        # # Test points are regularly spaced along [0,1]
        # # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_sampler = f3dasm.sampling.SobolSequence(design=self.data.design, seed=0)
            
            if function.dimensionality == 1:
                test_x = torch.linspace(0, 1, 500)
            else:
                test_x = torch.tensor(test_sampler.get_samples(numsamples=500).get_input_data().values)
            
            # observed_pred = surrogate.model.likelihood(surrogate.model(test_x))
            observed_pred = surrogate.predict(test_x)
            exact_y = function(test_x.numpy()[:, None])

        surrogate.plot_gpr(test_x=test_x, scaler=self.scaler, exact_y=exact_y, observed_pred=observed_pred, 
        train_x=torch.tensor(self.data.get_input_data().values), 
        train_y=torch.tensor(self.scaler.inverse_transform(self.data.get_output_data().values)))

    def update_step(self, function: Function) -> None:

        if self.parameter.acquisition_hyperparameters.best_f is not None:
            best_f_old = self.parameter.acquisition_hyperparameters.best_f
            best_f_new = np.amax(self.data.data['output'].values)

            self.parameter.acquisition_hyperparameters.best_f = np.amax([best_f_old, best_f_new])

        # if 'best_f' in self.parameter.acquisition_hyperparameters:

        #     best_f_old = self.parameter.acquisition_hyperparameters['best_f']
        #     best_f_new = np.amin(self.data.data['output'].values)

        #     self.parameter.acquisition_hyperparameters['best_f'] = np.amin([best_f_old, best_f_new])

        if len(self.data.data) == self.parameter.n_init:
            self.scaler = self.scale_output()

        regressor = self.parameter.regressor(
            train_data=self.data,
            design=self.data.design,
            parameter=self.parameter.regressor_hyperparameters,
        )

        surrogate = regressor.train()

        if self.parameter.visualize_gp:
            self.plot_gpr(surrogate, function)

        acquisition = self.parameter.acquisition(
            model=surrogate.model,
            **self.parameter.acquisition_hyperparameters.__dict__
        )

        # new_x, new_obj = optimize_acq_and_get_observation(acquisition, function)
        new_x, new_obj = optimize_acquisition(acquisition, function)

        self.data.add_numpy_arrays(input=new_x, output=self.scaler.transform(new_obj))

@dataclass
class MFBayesianOptimizationTorch_Parameters(OptimizerParameters):
    """Hyperparameters for MFBayesianOptimizationTorch optimizer"""

    regressor: f3dasm.Regressor = Stmf
    acquisition: Any = VFUpperConfidenceBound
    acquisition_hyperparameters: Any = None
    noise_fix: bool = False


@dataclass
class MFBayesianOptimizationTorch(Optimizer):
    """Bayesian optimization implementation from the botorch library"""
    multifidelity_function: MultiFidelityFunction = None

    def init_parameters(self) -> None:

        options = {
            "acquisition_hyperparameters": {
                "beta": 4,
                "maximize": False,
                },
        }

        self.parameter = MFBayesianOptimizationTorch_Parameters(**options)
        self.cost = 0

    def set_algorithm(self):
        self.algorithm = None

    def update_step_mf(self, multifidelity_function: MultiFidelityFunction, iteration: int,) -> None:

        if iteration == 0:
            self.data[-1].data = pd.concat([d.data for d in self.data], ignore_index=True)

        train_data = self.data[-1]

        regressor = Stmf(
            mf_train_data=train_data,
            mf_design=train_data.design,
            noise_fix=self.parameter.noise_fix,
        )

        surrogate = regressor.train()
        model = surrogate.model

        low_sampler = f3dasm.sampling.SobolSequence(design=self.data[0].design)
        high_sampler = f3dasm.sampling.SobolSequence(design=self.data[1].design)
        
        test_x_lf = low_sampler.get_samples(numsamples=500)
        test_x_hf = high_sampler.get_samples(numsamples=500)

        mean_high, _ = surrogate.predict(test_x_hf)
        mean_low, _ = surrogate.predict(test_x_lf)

        cr = multifidelity_function.costs[1] / multifidelity_function.costs[0]

        # self.parameter.acquisition = VFUpperConfidenceBound(
        #     model=model,
        #     beta=4,
        #     maximize=False,
        #     mean=[mean_low, mean_high],
        #     cr=cr,
        # )

        self.parameter.acquisition = VFUpperConfidenceBound(
            model=model,
            mean=[mean_low, mean_high],
            cr=cr,
            **self.parameter.acquisition_hyperparameters,
        )

        new_x, new_obj, cost = optimize_mfacq_and_get_observation(
            acq_f=self.parameter.acquisition,
            multifidelity_function=multifidelity_function,
            lf=multifidelity_function.fidelity_parameters[0],
            cost_model=cost_model
        )

        self.cost = cost
        print("This iteration cost:", float(self.cost))

        self.data[-1].add_numpy_arrays(input=new_x, output=new_obj)

def mf_data_compiler(
        mfdata: List[Data],
        fids: List[float],
) -> Data:
    for mfdata_level, fid in zip(mfdata, fids):
        mfdata_level.data['input', 'fid'] = fid * np.ones_like(mfdata_level.data['output'])
        mfdata_level.data = mfdata_level.data.sort_index(axis=1)

    mfdata[-1].data = pd.concat([mfdata_level.data for mfdata_level in mfdata])

    return mfdata[-1]