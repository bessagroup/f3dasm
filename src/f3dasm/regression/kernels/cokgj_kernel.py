#                                                                       Modules
# =============================================================================

# Standard
from typing import Optional

# Third-party
import GPy
import numpy as np
import torch
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from emukit.multi_fidelity.kernels import LinearMultiFidelityKernel
from gpytorch.kernels.kernel import Kernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from torch import Tensor

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Leo Guo (L.Guo@tudelft.nl)'
__credits__ = ['Leo Guo']
__status__ = 'Alpha'
# =============================================================================
#
# =============================================================================


tkwargs = {
    "dtype": torch.double,
    # "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "device": torch.device("cpu"),
}


class CoKrigingGP(SingleTaskGP):
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        outcome_transform: Optional[OutcomeTransform] = None,
        input_transform: Optional[InputTransform] = None,
        noise_fix: Optional[bool] = True,
    ) -> None:
        covar_module = ScaleKernel(CoKrigingKernel(noise_fix))
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            covar_module=covar_module,
            input_transform=input_transform,
            outcome_transform=outcome_transform,
        )


class CoKrigingKernel(Kernel):

    has_lengthscale = True

    def __init__(self, noise_fix=False, base_kernel=GPy.kern.RBF):
        super(CoKrigingKernel, self).__init__()
        self.noise_fix = noise_fix
        self.base_kernel = base_kernel

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        if self.noise_fix:
            kernels = [
                self.base_kernel(
                    x1.shape[-1] - 1, lengthscale=self.lengthscale.cpu().detach().numpy())
                + GPy.kern.White(x1.shape[-1] - 1),
                self.base_kernel(
                    x1.shape[-1] - 1, lengthscale=self.lengthscale.cpu().detach().numpy()),
            ]
        else:
            kernels = [
                self.base_kernel(
                    x1.shape[-1] - 1, lengthscale=self.lengthscale.cpu().detach().numpy())
                + GPy.kern.White(x1.shape[-1] - 1),
                self.base_kernel(
                    x1.shape[-1] - 1, lengthscale=self.lengthscale.cpu().detach().numpy())
                + GPy.kern.White(x1.shape[-1] - 1),
            ]
        lin_mf_kernel = LinearMultiFidelityKernel(kernels)

        x_len = len(np.shape(x1))
        if x_len == 4:
            x1 = x1[0][0]
            x2 = x2[0][0]
        elif x_len == 3:
            x1 = x1[0]
            x2 = x2[0]

        covar_cokg_np = lin_mf_kernel.K(
            x1.cpu().detach().numpy(), x2.cpu().detach().numpy()) + 1e-6

        if x_len == 4:
            covar_cokg_np = np.expand_dims(covar_cokg_np, axis=(0, 1))
        elif x_len == 3:
            covar_cokg_np = np.expand_dims(covar_cokg_np, axis=0)

        covar_cokg = torch.from_numpy(covar_cokg_np).to(**tkwargs)
        return covar_cokg


# Purgatory ###

# class CoKrigingKernel_DMS(*[Kernel]):
#
#     has_lengthscale = True
#
#     def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
#         kernels = [GPy.kern.RBF(x1.shape[-1] - 1, lengthscale=self.lengthscale.detach().numpy()),
#                    GPy.kern.RBF(x1.shape[-1] - 1, lengthscale=self.lengthscale.detach().numpy())]
#         lin_mf_kernel = LinearMultiFidelityKernel(kernels)
#
#         x_len = len(np.shape(x1))
#         if x_len == 4:
#             x1 = x1[0][0]
#             x2 = x2[0][0]
#         elif x_len == 3:
#             x1 = x1[0]
#             x2 = x2[0]
#
#         covar_cokg_np = lin_mf_kernel.K(x1.detach().numpy(), x2.detach().numpy()) + 1e-6
#
#         if x_len == 4:
#             covar_cokg_np = np.expand_dims(covar_cokg_np, axis=(0, 1))
#         elif x_len == 3:
#             covar_cokg_np = np.expand_dims(covar_cokg_np, axis=0)
#
#         covar_cokg = torch.from_numpy(covar_cokg_np)
#         return covar_cokg

# class NLCoKrigingKernel(Kernel):
#
#     has_lengthscale = True
#
#     def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
#         base_kernel = GPy.kern.RBF
#         kernels = make_non_linear_kernels(base_kernel, 2, X_train.shape[1] - 1)
#         nonlin_mf_model = NonLinearMultiFidelityModel(X_train, Y_train, n_fidelities=2, kernels=kernels,
#                                                       verbose=True, optimization_restarts=5)
#         covar_cokg = None
#         return covar_cokg


# class CoKrigingDMS(Model):
#     def __init__(self):
#         super().__init__()
#
#     def posterior(
#         self,
#         X: Tensor,
#         output_indices: Optional[List[int]] = None,
#         observation_noise: bool = False,
#         **kwargs: Any,
#     ) -> Posterior:
#         return CoKrigingDMSPosterior()
#
# class CoKrigingDMSPosterior(Posterior):
#     def __init__(self):
#         super().__init__()
#
#     def rsample(
#         self,
#         sample_shape: Optional[torch.Size] = None,
#         base_samples: Optional[Tensor] = None,
#     ) -> Tensor:
#         return torch.tensor(1)

# class CoKrigingGP_DMS(List[SingleTaskGP]):
#     def __init__(
#             # self,
#             # train_X: Tensor,
#             # train_Y: Tensor,
#             # # data_fidelity: Optional[int] = None,
#             # outcome_transform: Optional[OutcomeTransform] = None,
#             # input_transform: Optional[InputTransform] = None,
#             # prev_model: Optional[SingleTaskGP] = None,
#             self,
#             train_X: Tensor,
#             train_Y: Tensor,
#             # data_fidelity: Optional[int] = None,
#             outcome_transform: Optional[List[OutcomeTransform]] = None,
#             input_transform: Optional[List[InputTransform]] = None,
#             # prev_model: Optional[SingleTaskGP] = None,
#     ) -> None:
#         hf_volume = torch.sum(train_X[:, -1] == 1)
#         train_X_low, train_X_high = train_X[hf_volume:, :-1].detach().numpy(), train_X[:hf_volume, :-1].detach().numpy()
#         train_Y_low, train_Y_high = train_Y[hf_volume:].detach().numpy(), train_Y[:hf_volume].detach().numpy()
#         self.gpy_models = GPy.models.multiGPRegression([train_X_low, train_X_high], [train_Y_low, train_Y_high])
#
#         # k = GPy.kern.RBF(input_dim=len(train_X[0]))
#         # l = GPy.likelihoods.Gaussian()
#         # GP_level = GPy.core.gp.multiGP(
#         #     train_X, train_Y, model=prev_model,
#         #     kernel=k, likelihood=l
#         # )
#         super().__init__(
#             train_X=train_X,
#             train_Y=train_Y,
#             # likelihood=prev_model.likelihood,
#             # covar_module=prev_model.covar_module,
#             input_transform=input_transform,
#             outcome_transform=outcome_transform
#         )
#
#         # self.prev_model = prev_model
#         # self.rho = 2
#
#     def forward(self, x: Tensor) -> MultivariateNormal:
#
#         mean_x, _ = self.gpy_models.predict(x)
#         covar_x = self.gpy_models[0].posterior.covariance
#
#         # if self.training:
#         #     x = self.transform_inputs(x)
#         #
#         # mean_x = self.mean_module(x)
#         # covar_x = self.covar_module(x)
#         #
#         # mean_x_prev = self.prev_model.mean_module(x)
#         # covar_x_prev = self.prev_model.covar_module(x)
#         #
#         # mean_x = mean_x + self.rho * mean_x_prev
#         # covar_x = covar_x + self.rho ** 2 * covar_x_prev
#
#         return MultivariateNormal(mean_x, covar_x)

# class NLCoKrigingGP(SingleTaskGP):
#     def __init__(
#             self,
#             train_X: Tensor,
#             train_Y: Tensor,
#             # data_fidelity: Optional[int] = None,
#             outcome_transform: Optional[OutcomeTransform] = None,
#             input_transform: Optional[InputTransform] = None,
#     ) -> None:
#         super().__init__(
#             train_X=train_X,
#             train_Y=train_Y,
#             input_transform=input_transform,
#             outcome_transform=outcome_transform
#         )
#
#     def forward(self, x: Tensor) -> MultivariateNormal:
#         if self.training:
#             x = self.transform_inputs(x)
#
#         base_kernel = GPy.kern.RBF
#         kernels = make_non_linear_kernels(base_kernel, 2, x.detach().numpy().shape[-1] - 1)
#
#         train_X = self.train_inputs[0].detach().numpy()[:, :-1]
#         fid = self.train_inputs[0].detach().numpy()[:, -1]
#         fid = (fid[:, None] - min(fid)) / (max(fid) - min(fid))
#         train_X = np.hstack((train_X, fid))
#
#         train_Y = self.train_targets.detach().numpy()[:, None]
#
#         nonlin_mf_model = NonLinearMultiFidelityModel(train_X,
#                                                       train_Y,
#                                                       n_fidelities=2, kernels=kernels,
#                                                           verbose=False, optimization_restarts=5)
#
#         mean_x, covar_x = nonlin_mf_model.predict(x.detach().numpy())
#
#         # Does not work... distribution is not Gaussian
#         return MultivariateNormal(torch.from_numpy(mean_x), torch.from_numpy(covar_x))
