#                                                                       Modules
# =============================================================================

# Standard
from dataclasses import dataclass
from typing import Any, Tuple

# Third-party
import torch
from botorch.acquisition import (InverseCostWeightedUtility,
                                 UpperConfidenceBound)
from botorch.models import AffineFidelityCostModel
from botorch.optim import optimize_acqf, optimize_acqf_mixed

# Locals
from .. import Function, Optimizer, OptimizerParameters
from ..regression.gpr import Sogpr

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Leo Guo', 'Martin van der Schelling']
__status__ = 'Alpha'
# =============================================================================
#
# =============================================================================


@dataclass
class BayesianOptimizationTorch_Parameters(OptimizerParameters):
    """Hyperparameters for BayesianOptimization optimizer"""

    model: Any = None
    acquisition: Any = None


def cost_model(self):
    lf = self.fidelities[0]
    a = (1 - 1 / self.cost_ratio) / (1 - lf)
    cost_model = AffineFidelityCostModel(
        fidelity_weights={self.objective_function.dim - 1: a}, fixed_cost=1 - a)
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


def optimize_mfacq_and_get_observation(acq_f, cost_model, lf, function: Function) -> Tuple[Any, Any, Any]:
    fixed_features_list = [{-1: float(lf)}, {-1: 1.0}]
    candidates, _ = optimize_acqf_mixed(
        acq_function=acq_f,
        bounds=torch.tensor(function.input_domain.T, dtype=torch.float64),
        fixed_features_list=fixed_features_list,
        q=1,
        num_restarts=10,
        raw_samples=15,
        options={"batch_limit": 1, "maxiter": 200},
        sequential=True,
    )
    cost = cost_model()(candidates).sum()
    new_x = candidates.numpy()
    new_obj = function(new_x)
    return new_x, new_obj, cost


class BayesianOptimizationTorch(Optimizer):
    """Bayesian optimization implementation from the botorch library"""

    def init_parameters(self):

        train_x = self.data.data["input"].values
        train_y = self.data.data["output"].values

        regressor = Sogpr(
            train_input_data=train_x,
            train_output_data=train_y,
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

    def set_algorithm(self):
        self.algorithm = None

    def update_step(self, function: Function):
        new_x, new_obj = optimize_acq_and_get_observation(
            self.parameter.acquisition, function)

        self.data.add_numpy_arrays(input=new_x, output=new_obj)

        train_x = self.data.data["input"].values
        train_y = self.data.data["output"].values

        regressor = Sogpr(
            train_input_data=train_x,
            train_output_data=train_y,
            design=self.data.design,
        )

        model = regressor.train().model

        self.parameter.acquisition = UpperConfidenceBound(
            model=model,
            beta=4,
            maximize=False,
        )

        # self.parameter.acquisition = ExpectedImprovement(
        #     model=model,
        #     best_f=torch.amin(torch.tensor(train_y)),
        #     maximize=False,
        # )


class MFBayesianOptimizationTorch(Optimizer):
    """Bayesian optimization implementation from the botorch library"""

    def init_parameters(self):

        train_x = self.data.data["input"].values
        train_y = self.data.data["output"].values

        regressor = Sogpr(
            train_input_data=train_x,
            train_output_data=train_y,
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

    def set_algorithm(self):
        self.algorithm = None

    def update_step(self, function: Function):
        new_x, new_obj = optimize_acq_and_get_observation(
            self.parameter.acquisition, function)

        self.data.add_numpy_arrays(input=new_x, output=new_obj)

        train_x = self.data.data["input"].values
        train_y = self.data.data["output"].values

        regressor = Sogpr(
            train_input_data=train_x,
            train_output_data=train_y,
            design=self.data.design,
        )

        model = regressor.train().model

        self.parameter.acquisition = UpperConfidenceBound(
            model=model,
            beta=4,
            maximize=False,
        )

        # self.parameter.acquisition = ExpectedImprovement(
        #     model=model,
        #     best_f=torch.amin(torch.tensor(train_y)),
        #     maximize=False,
        # )
