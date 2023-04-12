#                                                                       Modules
# =============================================================================

# Standard
from dataclasses import dataclass, field
from typing import Any, Tuple, List

# Third-party core
import scipy
import numpy as np
import pandas as pd

# Local
from .._imports import try_import
from .optimizer import Optimizer, OptimizerParameters, MultiFidelityOptimizer
from ..design.experimentdata import ExperimentData
from ..sampling import SobolSequence_torch
from ..functions import Function, MultiFidelityFunction
from ..machinelearning.acquisition_functions import VFUpperConfidenceBound, UpperConfidenceBound
from ..machinelearning.gpr import Cokgj, Sogpr, MultitaskGPR, Sogpr_Parameters, Cokgj_Parameters

# Third-party extension
with try_import(['optimization', 'machinelearning']) as _imports:
    import torch
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler

import gpytorch

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Leo Guo (L.L.Guo@tudelft.nl)'
__credits__ = ['Leo Guo', 'Martin van der Schelling']
__status__ = 'Alpha'
# =============================================================================
#
# =============================================================================


@dataclass
class Acquisition_Parameters():
    """Hyperparameters for Acquisition function"""
    
    best_f: Any = None
    beta: float = 0.4
    maximize: bool = False


@dataclass
class BayesianOptimizationTorch_Parameters(OptimizerParameters):
    """Hyperparameters for BayesianOptimizationTorch optimizer"""

    regressor: Any = Sogpr
    acquisition: Any = UpperConfidenceBound
    regressor_hyperparameters: Sogpr_Parameters = Sogpr_Parameters()
    acquisition_hyperparameters: Acquisition_Parameters = Acquisition_Parameters()
    n_init: int = 10
    visualize_gp: bool = False


def optimize_acquisition(acq_f, function, dimensionality=None,):
    if dimensionality is None:
        dimensionality = function.dimensionality
    res = scipy.optimize.minimize(
        fun=lambda x: -acq_f(torch.tensor(x[None, :])),
        x0=np.random.rand(dimensionality),
        method='Nelder-Mead',
        bounds=scipy.optimize.Bounds(lb=function.scale_bounds[:, 0], ub=function.scale_bounds[:, 1]),
        options={'disp': True, 'maxfev': 5}#, 'eps': 1e-4, 'maxiter': 999, 'maxfun': 1},
     )
    new_x = res.x[None, :]
    acq_val = -res.fun
    new_obj = function(new_x)
    return new_x, new_obj, acq_val


def optimize_multifidelity_acquisition(multifidelity_acq_f, multifidelity_function, dimensionality):
    acq_val_tot = -torch.inf
    for fid, cost in enumerate(multifidelity_function.costs):
        acq_f = lambda x: multifidelity_acq_f(torch.tensor(x[:, None]), fid=fid)

        new_x, new_obj, acq_val = optimize_acquisition(acq_f, multifidelity_function.fidelity_functions[fid], dimensionality)
        if acq_val > acq_val_tot:
            new_x_tot = new_x
            new_obj_tot = new_obj
            acq_val_tot = acq_val
            cost_tot = cost
            fid_tot = fid
    return new_x_tot, new_obj_tot, cost_tot, fid_tot


@dataclass
class BayesianOptimizationTorch(Optimizer):
    """Bayesian optimization implementation based on the gpytorch library"""

    parameter: BayesianOptimizationTorch_Parameters = BayesianOptimizationTorch_Parameters()

    def set_algorithm(self):
        self.algorithm = None

    def scale_output(self):
        scaler = StandardScaler()
        self.data.data['output'] = scaler.fit_transform(self.data.data['output'])
        return scaler

    def plot_gpr(self, surrogate, function, acq=None,):
        # Get into evaluation (predictive posterior) mode
        surrogate.model.eval()
        surrogate.model.likelihood.eval()

        # # Test points are regularly spaced along [0,1]
        # # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            test_sampler = SobolSequence_torch(design=self.data.design, seed=0)
            
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

        if acq is not None:
            acq_plot = torch.tensor([acq(x[:, None]) for x in test_x[:, None]])
            plt.figure()
            plt.plot(test_x, acq_plot)

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

        acquisition = self.parameter.acquisition(
            model=surrogate.model,
            **self.parameter.acquisition_hyperparameters.__dict__
        )

        if self.parameter.visualize_gp:
            self.plot_gpr(surrogate, function, acq=acquisition)

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

    def plot_gpr(self, surrogate, multifidelity_function, acq=None,):
        # Get into evaluation (predictive posterior) mode
        surrogate.model.eval()
        surrogate.model.likelihood.eval()

        dim = multifidelity_function.fidelity_functions[0].dimensionality

        for i in range(len(multifidelity_function.fidelity_functions)):
            # # Test points are regularly spaced along [0,1]
            # # Make predictions by feeding model through likelihood
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                test_sampler = SobolSequence_torch(design=self.data[0].design, seed=0)
                
                if dim == 1:
                    test_x = torch.linspace(0, 1, 50)
                else:
                    test_x = torch.tensor(test_sampler.get_samples(numsamples=500).get_input_data().values)
                
                # observed_pred = surrogate.model.likelihood(surrogate.model(test_x))
                
                if i == 0:
                    test_x_pad = [test_x, torch.empty(0, dim)]
                else:
                    test_x_pad = [torch.empty(0, dim), test_x]
                
                observed_pred = surrogate.predict(test_x_pad)
                exact_y = multifidelity_function.fidelity_functions[i](test_x.numpy()[:, None])
            
            plt.figure(num='gp')
            plt.plot(test_x, observed_pred.mean)
            plt.plot(self.data[i].get_input_data().values, self.data[i].get_output_data().values, '*')

            # surrogate.plot_gpr(test_x=test_x, scaler=self.scaler, exact_y=exact_y, observed_pred=observed_pred, 
            # train_x=torch.tensor(self.data.get_input_data().values), 
            # train_y=torch.tensor(self.scaler.inverse_transform(self.data.get_output_data().values)))

            if acq is not None:
                acq_plot = torch.tensor([acq(x[:, None], fid=i) for x in test_x[:, None]])
                index = torch.argmax(acq_plot)
                plt.figure('Acq')
                p = plt.plot(test_x, acq_plot, label='fidelity ' + str(i))
                plt.plot(test_x[index], acq_plot[index], '*', color=p[0].get_color(), label='Max. acquisition at fidelity ' + str(i))
        
        plt.xlabel('design space')
        plt.ylabel('belief')
        plt.tight_layout()
        plt.legend()

    def set_scaler(self):
        scaler = StandardScaler()
        scaler.fit(self.data[0].get_output_data().values)
        self.scaler = scaler

    def update_step(self, multifidelity_function: MultiFidelityFunction,) -> None:

        self.set_scaler()
        for i in range(2):
            scaled_output = self.scaler.transform(self.data[i].get_output_data())
            self.data[i].add_output(scaled_output)

        design = self.data[0].design
        self.dimensionality = len(design.input_space)

        regressor = Cokgj(
            train_data=self.data, 
            design=design,
            parameter=self.parameter.regressor_hyperparameters,
        )

        surrogate = regressor.train()

        sampler = SobolSequence_torch(design=design)
        
        if self.dimensionality == 1:
            test_x = torch.linspace(0, 1, 50)
        else:
            test_x = sampler.get_samples(numsamples=500).get_input_data().values

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred_hf = surrogate.predict([torch.empty(0, self.dimensionality), 
                torch.tensor(test_x)])
            observed_pred_lf = surrogate.predict([torch.tensor(test_x), 
                torch.empty(0, self.dimensionality)])

        acquisition = self.parameter.acquisition(
            model=surrogate.model,
            mean=[observed_pred_lf.mean, observed_pred_hf.mean],
            cr=multifidelity_function.costs[1] / multifidelity_function.costs[0],
            **self.parameter.acquisition_hyperparameters.__dict__,
        )

        if self.parameter.visualize_gp:
            f, axs = plt.subplots(2, 1, num='gp and acq', figsize=(6, 8), sharex='all')
            # self.plot_gpr(surrogate, multifidelity_function, acq=acquisition)

            surrogate.plot_gpr(
                test_x=test_x,
                exact_y=None, 
                observed_pred=observed_pred_hf,
                train_x=torch.tensor(self.data[1].get_input_data().values),
                train_y=torch.tensor(self.scaler.inverse_transform(self.data[1].get_output_data().values)),
                color='b',
                acquisition=acquisition,
                fid=1,
                axs=axs,
                scaler=self.scaler
                )
            
            surrogate.plot_gpr(
                test_x=test_x,
                exact_y=None, 
                observed_pred=observed_pred_lf,
                train_x=torch.tensor(self.data[0].get_input_data().values),
                train_y=torch.tensor(self.scaler.inverse_transform(self.data[0].get_output_data().values)),
                color='orange',
                acquisition=acquisition,
                fid=0,
                axs=axs,
                scaler=self.scaler
                )
                
            axs[0].set_ylabel('y')
            axs[1].set_xlabel('x')
            axs[1].set_ylabel('acquisition')

            plt.tight_layout()

        new_x, new_obj, cost, fid = optimize_multifidelity_acquisition(
            multifidelity_acq_f=acquisition,
            multifidelity_function=multifidelity_function,
            dimensionality=self.dimensionality,
        )

        return new_x, self.scaler.transform(new_obj), cost, fid


def mf_data_compiler(
        mfdata: List[ExperimentData],
        fids: List[float],
) -> ExperimentData:
    for mfdata_level, fid in zip(mfdata, fids):
        mfdata_level.data['input', 'fid'] = fid * np.ones_like(mfdata_level.data['output'])
        mfdata_level.data = mfdata_level.data.sort_index(axis=1)

    mfdata[-1].data = pd.concat([mfdata_level.data for mfdata_level in mfdata])

    return mfdata[-1]