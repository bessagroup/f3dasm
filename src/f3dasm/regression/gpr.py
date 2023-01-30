# from typing import List
from dataclasses import dataclass

import gpytorch.kernels
from botorch.models import MultiTaskGP, SingleTaskMultiFidelityGP
from botorch.models.gp_regression import SingleTaskGP
from gpytorch.kernels import ScaleKernel, RBFKernel

from gpytorch.models import ExactGP

import torch

# from f3dasm.base.data import Data

# from ..base.regression import Regressor, Surrogate
from .adapters.torch_implementations import TorchGPRegressor, TorchGPSurrogate
from .kernels import cokgj_kernel
from .kernels.cokgj_kernel import CoKrigingGP

# import torch

# # import sys
# # sys.path.insert(0, 'GPy')
# import GPy

# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):

    num_outputs = 1

    def __init__(self, train_x, train_y, likelihood, mean_module, covar_module):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module
        self.covar_module = covar_module
    
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

@dataclass
class Sogpr_Parameters:
    """(Pre-initialized) hyperparameters for single-fidelity Gaussian process regression in pytorch"""

    likelihood: gpytorch.likelihoods.Likelihood = gpytorch.likelihoods.GaussianLikelihood()
    mean: gpytorch.means.Mean = gpytorch.means.ZeroMean()
    kernel: gpytorch.kernels.Kernel = ScaleKernel(RBFKernel())
    noise_fix: bool = False
    opt_algo: torch.optim.Optimizer = torch.optim.Adam
    verbose_training: bool = False
    training_iter: int = 50


class Sogpr(TorchGPRegressor):
    def __init__(
        self,
        regressor=ExactGPModel,#SingleTaskGP,
        parameter: Sogpr_Parameters = Sogpr_Parameters(),
        train_data=None,
        design=None,
        # noise_fix: bool = False
    ):

        super().__init__(
            regressor=regressor,
            parameter=parameter,
            train_data=train_data,
            # covar_module=parameter.kernel,
            # mean_module=parameter.mean,
            design=design,
            # noise_fix=noise_fix,
        )

        # self.regressor = regressor
        # self.mean = parameter.mean
        # self.kernel = parameter.kernel


@dataclass
class Mtask_Parameters:
    kernel: gpytorch.kernels.Kernel = ScaleKernel(RBFKernel())


class Mtask(TorchGPRegressor):
    def __init__(
        self,
        regressor=MultiTaskGP,
        parameter=Mtask_Parameters(),
        mf_train_data=None,
        mf_design=None,
    ):
        super().__init__(
            train_data=mf_train_data,
            covar_module=parameter.kernel,
            design=mf_design,
            task_feature=-1,
        )

        self.regressor = regressor
        self.kernel = parameter.kernel


@dataclass
class Cokgj_Parameters:
    kernel: gpytorch.kernels.Kernel = cokgj_kernel.CoKrigingKernel()


class Cokgj(TorchGPRegressor):
    def __init__(
        self,
        regressor=CoKrigingGP,
        parameter=Cokgj_Parameters(),
        mf_train_data=None,
        mf_design=None,
    ):
        super().__init__(
            train_data=mf_train_data,
            design=mf_design,
        )

        self.regressor = regressor
        self.kernel = parameter.kernel


@dataclass
class Stmf_Parameters:
    linear_truncated: bool = False
    data_fidelity: int = -1

class Stmf(TorchGPRegressor):
    def __init__(
        self,
        regressor=SingleTaskMultiFidelityGP,
        parameter=Stmf_Parameters(),
        mf_train_data=None,
        mf_design=None,
        noise_fix: bool = False
    ):
        super().__init__(
            train_data=mf_train_data,
            linear_truncated=parameter.linear_truncated,
            data_fidelity=parameter.data_fidelity,
            design=mf_design,
            noise_fix=noise_fix,
        )

        self.regressor = regressor

# @dataclass
# class Cokgd_Parameters:
#     pass

# class Cokgd(Regressor):
    
#     def train(self) -> Surrogate:
        
#         base_k = GPy.kern.RBF
#         kernels_RL = [base_k(dim - 1) + GPy.kern.White(dim - 1), base_k(dim - 1)]
#         model = GPy.models.multiGPRegression(
#             self.train_data['input'],
#             self.train_data['output'],
#             kernel=kernels_RL,
#         )
        
#         return model