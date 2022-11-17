from dataclasses import dataclass

import gpytorch.kernels
from botorch.models import MultiTaskGP, SingleTaskMultiFidelityGP
from botorch.models.gp_regression import SingleTaskGP
from gpytorch.kernels import ScaleKernel, RBFKernel

from .adapters.torch_implementations import TorchGPRegressor
from .kernels import cokgj_kernel
from .kernels.cokgj_kernel import CoKrigingGP


@dataclass
class Sogpr_Parameters:
    kernel: gpytorch.kernels.Kernel = ScaleKernel(RBFKernel())


class Sogpr(TorchGPRegressor):
    def __init__(
        self,
        regressor=SingleTaskGP,
        parameter=Sogpr_Parameters(),
        train_data=None,
        design=None,
    ):

        super().__init__(
            train_data=train_data,
            covar_module=parameter.kernel,
            design=design,
        )

        self.regressor = regressor
        self.kernel = parameter.kernel


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
    ):
        super().__init__(
            train_data=mf_train_data,
            linear_truncated=parameter.linear_truncated,
            data_fidelity=parameter.data_fidelity,
            design=mf_design,
        )

        self.regressor = regressor
