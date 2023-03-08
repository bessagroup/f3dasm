#                                                                       Modules
# =============================================================================

# Standard
from dataclasses import dataclass, field
from typing import Any, Tuple, List

# Third-party
import scipy
import numpy as np
import pandas as pd
import torch
from botorch.acquisition import (InverseCostWeightedUtility,
                                 UpperConfidenceBound)
from botorch.models import AffineFidelityCostModel
from botorch.optim import optimize_acqf, optimize_acqf_mixed

# Locals
# from .. import Optimizer, Function, OptimizerParameters, Data, MultiFidelityFunction
from ..optimization.optimizer import Optimizer, OptimizerParameters, MultiFidelityOptimizer
from ..design.experimentdata import ExperimentData
from ..base.function import Function, MultiFidelityFunction
from ..base.acquisition import VFUpperConfidenceBound, UpperConfidenceBound
# from ..regression.gpr import Cokgj, Sogpr, MultitaskGPR, Sogpr_Parameters
from ..machinelearning.gpr import Cokgj, Sogpr, MultitaskGPR, Sogpr_Parameters, Cokgj_Parameters
import f3dasm
from sklearn.preprocessing import StandardScaler

import gpytorch

@dataclass
class Acquisition_Parameters():
    """Hyperparameters for Acquisition function"""
    
    best_f: Any = None
    beta: float = 0.4
    maximize: bool = False

# from .adapters.optimization import Optimizer

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Leo Guo (L.L.Guo@tudelft.nl)'
__credits__ = ['Leo Guo', 'Martin van der Schelling']
__status__ = 'Alpha'
# =============================================================================
#
# =============================================================================


@dataclass
class BayesianOptimizationTorch_Parameters(OptimizerParameters):
    """Hyperparameters for BayesianOptimizationTorch optimizer"""

    regressor: Any = Sogpr
    acquisition: Any = UpperConfidenceBound
    regressor_hyperparameters: Sogpr_Parameters = Sogpr_Parameters()
    acquisition_hyperparameters: Acquisition_Parameters = Acquisition_Parameters()
    n_init: int = 10
    visualize_gp: bool = False


# def cost_model(lf, cost_ratio):
#     a = (1 - 1 / cost_ratio) / (1 - lf)
#     cost_model = AffineFidelityCostModel(
#         fidelity_weights={-1: a},
#         fixed_cost=1 - a
#     )
#     return cost_model


# def cost_aware_utility(self):
#     return InverseCostWeightedUtility(cost_model=self.cost_model())


# def optimize_acq_and_get_observation(acq_f, function: Function) -> Tuple[Any, Any]:
#     candidates, _ = optimize_acqf(
#         acq_function=acq_f,
#         # bounds=torch.tensor(function.input_domain.T, dtype=torch.float64),
#         bounds=torch.tile(torch.tensor([[0.0], [1.0]]), (1, function.dimensionality)),
#         q=1,
#         num_restarts=10,
#         raw_samples=15,
#     )
#     new_x = candidates.numpy()
#     new_obj = function(new_x)
#     return new_x, new_obj


# def optimize_mfacq_and_get_observation(acq_f, cost_model, lf, multifidelity_function: MultiFidelityFunction) -> Tuple[Any, Any, Any]:

#     dim = multifidelity_function.fidelity_functions[0].dimensionality
#     fixed_features_list = [{dim: float(lf)},
#                            {dim: 1.0}]

#     # design_bounds = torch.tensor(base_fun.input_domain.T, dtype=torch.float64)
#     design_bounds = torch.tile(torch.tensor([[0.0], [1.0]]), (1, dim))
#     fidelity_bounds = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
#     augmented_bounds = torch.hstack((design_bounds, fidelity_bounds))

#     candidates, _ = optimize_acqf_mixed(
#         acq_function=acq_f,
#         bounds=augmented_bounds,
#         fixed_features_list=fixed_features_list,
#         q=1,
#         num_restarts=10,
#         raw_samples=15,
#         options={"batch_limit": 1, "maxiter": 200},
#         sequential=True,
#     )

#     cost = cost_model(lf, cost_ratio=2)(candidates).sum()
#     new_x = candidates.numpy()
#     function = multifidelity_function.get_fidelity_function_by_parameter(fidelity_parameter=new_x[0, -1])
#     new_obj = function(new_x)

#     print(new_x, new_obj)

#     return new_x, new_obj, cost


def optimize_acquisition(acq_f, function):
    res = scipy.optimize.minimize(
        fun=lambda x: -acq_f(torch.tensor(x[None, :])),
        x0=np.random.rand(function.dimensionality),
        method='L-BFGS-B', 
        bounds=scipy.optimize.Bounds(lb=function.scale_bounds[:, 0], ub=function.scale_bounds[:, 1]),
        options={'disp': False, 'eps': 1e-4},
     )
    new_x = res.x[None, :]
    acq_val = -res.fun
    new_obj = function(new_x)
    return new_x, new_obj, acq_val

def optimize_multifidelity_acquisition(multifidelity_acq_f, multifidelity_function, test_x_hf):
    acq_val_tot = -torch.inf
    for fid, cost in enumerate(multifidelity_function.costs):
        acq_f = lambda x: multifidelity_acq_f(torch.tensor(x[:, None]), fid=fid)

        new_x, new_obj, acq_val = optimize_acquisition(acq_f, multifidelity_function.fidelity_functions[fid])
        if acq_val > acq_val_tot:
            new_x_tot = new_x
            new_obj_tot = new_obj
            acq_val_tot = acq_val
            cost_tot = cost
            fid_tot = fid
    return new_x_tot, new_obj_tot, cost_tot, fid_tot

class BayesianOptimizationTorch(Optimizer):
    """Bayesian optimization implementation based on the gpytorch library"""

    parameter: BayesianOptimizationTorch_Parameters = BayesianOptimizationTorch_Parameters()

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

    def update_step(self, function: Function) -> Tuple:

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
        new_x, new_obj, _ = optimize_acquisition(acquisition, function)

        # self.data.add_numpy_arrays(input=new_x, output=self.scaler.transform(new_obj))

        return new_x, self.scaler.transform(new_obj)

@dataclass
class MFBayesianOptimizationTorch_Parameters(OptimizerParameters):
    """Hyperparameters for MFBayesianOptimizationTorch optimizer"""

    regressor: Any = MultitaskGPR
    acquisition: Any = VFUpperConfidenceBound

    regressor_hyperparameters: Any = Cokgj_Parameters()
    acquisition_hyperparameters: Acquisition_Parameters = Acquisition_Parameters()
    n_init: int = 10
    visualize_gp: bool = False


@dataclass
class MFBayesianOptimizationTorch(MultiFidelityOptimizer):
    """Bayesian optimization implementation from the botorch library"""
    multifidelity_function: MultiFidelityFunction = None
    parameter: MFBayesianOptimizationTorch_Parameters = MFBayesianOptimizationTorch_Parameters()

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

    def update_step(self, multifidelity_function: MultiFidelityFunction,) -> None:

        regressor = f3dasm.machinelearning.gpr.Cokgj(
            mf_train_data=self.data, 
            design=self.data[0].design,
            parameter=self.parameter.regressor_hyperparameters,
        )

        surrogate = regressor.train()

        low_sampler = f3dasm.sampling.SobolSequence(design=self.data[0].design)
        high_sampler = f3dasm.sampling.SobolSequence(design=self.data[1].design)
        
        test_x_lf = low_sampler.get_samples(numsamples=500)
        test_x_hf = high_sampler.get_samples(numsamples=500)

        # mean_high = surrogate.predict([torch.tensor([])[:, None], torch.tensor(test_x_hf.get_input_data().values)]).mean
        mean_high = surrogate.predict([torch.empty(0, multifidelity_function.fidelity_functions[0].dimensionality), 
            torch.tensor(test_x_hf.get_input_data().values)]).mean
        mean_low = surrogate.predict([torch.tensor(test_x_lf.get_input_data().values), 
            torch.empty(0, multifidelity_function.fidelity_functions[0].dimensionality)]).mean

        cr = multifidelity_function.costs[1] / multifidelity_function.costs[0]

        # self.parameter.acquisition = VFUpperConfidenceBound(
        #     model=model,
        #     beta=4,
        #     maximize=False,
        #     mean=[mean_low, mean_high],
        #     cr=cr,
        # )

        acquisition = self.parameter.acquisition(
            model=surrogate.model,
            mean=[mean_low, mean_high],
            cr=cr,
            **self.parameter.acquisition_hyperparameters.__dict__,
        )

        new_x, new_obj, cost, fid = optimize_multifidelity_acquisition(
            multifidelity_acq_f=acquisition,
            multifidelity_function=multifidelity_function,
            test_x_hf=test_x_hf,
        )

        # self.cost = cost
        # print("This iteration cost:", float(self.cost))

        # self.data[fid].add_numpy_arrays(input=new_x, output=new_obj)

        return new_x, new_obj, cost, fid

def mf_data_compiler(
        mfdata: List[ExperimentData],
        fids: List[float],
) -> ExperimentData:
    for mfdata_level, fid in zip(mfdata, fids):
        mfdata_level.data['input', 'fid'] = fid * np.ones_like(mfdata_level.data['output'])
        mfdata_level.data = mfdata_level.data.sort_index(axis=1)

    mfdata[-1].data = pd.concat([mfdata_level.data for mfdata_level in mfdata])

    return mfdata[-1]