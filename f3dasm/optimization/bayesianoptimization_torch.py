from dataclasses import dataclass
from typing import Any, Tuple, List

import numpy as np
import pandas as pd
import torch
from botorch.acquisition import InverseCostWeightedUtility, UpperConfidenceBound
from botorch.models import AffineFidelityCostModel
from botorch.optim import optimize_acqf, optimize_acqf_mixed

from .. import Optimizer, Function, OptimizerParameters, Data, MultiFidelityFunction
from ..base.acquisition import VFUpperConfidenceBound
from ..regression.gpr import Sogpr, Stmf
import f3dasm


@dataclass
class BayesianOptimizationTorch_Parameters(OptimizerParameters):
    """Hyperparameters for BayesianOptimization optimizer"""

    model: Any = None
    acquisition: Any = None


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
        bounds=torch.tensor(function.input_domain.T, dtype=torch.float64),
        q=1,
        num_restarts=10,
        raw_samples=15,
    )
    new_x = candidates.numpy()
    new_obj = function(new_x)
    return new_x, new_obj


def optimize_mfacq_and_get_observation(acq_f, cost_model, lf, mffunction: MultiFidelityFunction) -> Tuple[Any, Any, Any]:

    fixed_features_list = [{0: float(lf)},
                           {0: 1.0}]

    base_fun = mffunction.funs[0].base_fun

    design_bounds = torch.tensor(base_fun.input_domain.T, dtype=torch.float64)
    fidelity_bounds = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    augmented_bounds = torch.hstack((fidelity_bounds, design_bounds))

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

    cost = cost_model(lf, cost_ratio=10)(candidates).sum()
    new_x = candidates.numpy()
    function = mffunction.get_fun_by_fid(fid=lf)
    new_obj = function(new_x)

    return new_x, new_obj, cost


class BayesianOptimizationTorch(Optimizer):
    """Bayesian optimization implementation from the botorch library"""

    def init_parameters(self) -> None:

        regressor = Sogpr(
            train_data=self.data,
            design=self.data.design,
        )

        model = regressor.train().model

        # acquisition = ExpectedImprovement(
        #     model=model,
        #     best_f=torch.amin(torch.tensor(train_y)),
        #     maximize=False,
        # )

        acquisition = UpperConfidenceBound(
            model=model,
            beta=4,
            maximize=False,
        )

        options = {
            "model": model,
            "acquisition": acquisition,
        }

        self.parameter = BayesianOptimizationTorch_Parameters(**options)

    def set_algorithm(self) -> None:
        self.algorithm = None

    def update_step(self, function: Function) -> None:

        regressor = Sogpr(
            train_data=self.data,
            design=self.data.design,
        )

        model = regressor.train().model

        self.parameter.acquisition = UpperConfidenceBound(
            model=model,
            beta=4,
            maximize=False,
        )

        new_x, new_obj = optimize_acq_and_get_observation(self.parameter.acquisition, function)

        self.data.add_numpy_arrays(input=new_x, output=new_obj)

        # self.parameter.acquisition = ExpectedImprovement(
        #     model=model,
        #     best_f=torch.amin(torch.tensor(train_y)),
        #     maximize=False,
        # )

@dataclass

class MFBayesianOptimizationTorch(Optimizer):
    """Bayesian optimization implementation from the botorch library"""
    mffun: MultiFidelityFunction = None

    def init_parameters(self) -> None:
        # train_x_space = self.data.data.iloc[:, 1:-1].values
        # train_x_fid = self.data.data["input", 'fid'].values[:, None]
        # train_x = np.hstack((train_x_space, train_x_fid))
        # train_y = self.data.data["output"].values
        #
        # regressor = Stmf(
        #     mf_train_input_data=train_x,
        #     mf_train_output_data=train_y,
        #     mf_design=self.data.design,
        # )
        #
        # model = regressor.train().model
        #
        # acquisition = VFUpperConfidenceBound(
        #     model=model,
        #     beta=4,
        #     maximize=False,
        # )

        options = {
            "model": Stmf,
            "acquisition": VFUpperConfidenceBound,
        }

        self.parameter = BayesianOptimizationTorch_Parameters(**options)

    def set_algorithm(self) -> None:
        self.algorithm = None

    def update_step_mf(self, mffunction: MultiFidelityFunction, iteration: int,) -> None:

        if iteration == 0:
            self.data[-1].data = pd.concat([d.data for d in self.data], ignore_index=True)

        train_data = self.data[-1]

        regressor = Stmf(
            mf_train_data=train_data,
            mf_design=train_data.design,
        )

        surrogate = regressor.train()
        model = surrogate.model

        low_sampler = f3dasm.sampling.SobolSequence(design=self.data[0].design)
        high_sampler = f3dasm.sampling.SobolSequence(design=self.data[1].design)
        
        test_x_lf = low_sampler.get_samples(numsamples=500)
        test_x_hf = high_sampler.get_samples(numsamples=500)

        mean_high, _ = surrogate.predict(test_x_hf)
        mean_low, _ = surrogate.predict(test_x_lf)

        self.parameter.acquisition = VFUpperConfidenceBound(
            model=model,
            beta=4,
            maximize=False,
            mean=[mean_low, mean_high]
        )

        new_x, new_obj, cost = optimize_mfacq_and_get_observation(
            acq_f=self.parameter.acquisition,
            mffunction=mffunction,
            lf=0.5,
            cost_model=cost_model
        )

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