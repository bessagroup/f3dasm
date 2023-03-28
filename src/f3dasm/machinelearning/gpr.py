#                                                                       Modules
# =============================================================================

# Standard
from dataclasses import dataclass, field
import warnings
from typing import Any, List
import functools
import math

# Local
from .._imports import try_import
from .adapters.torch_implementations import TorchGPRegressor

# Third-party extension
with try_import('machinelearning') as _imports:
    import torch
    import gpytorch
    from gpytorch.models import ExactGP as GPyTorch_ExactGP
    from gpytorch.models.exact_prediction_strategies import DefaultPredictionStrategy as GPyTorch_PredictionStrategy
    import linear_operator

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Leo Guo (L.L.Guo@tudelft.nl)'
__credits__ = ['Leo Guo']
__status__ = 'Alpha'
# =============================================================================
#
# =============================================================================

if not _imports.is_successful():
    GPyTorch_ExactGP = object # NOQA

# We will use the simplest form of GP model, exact inference
class ExactGPModel(GPyTorch_ExactGP):

    num_outputs = 1

    def __init__(self, train_x, train_y, likelihood, mean_module, covar_module):
        _imports.check()
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
    kernel: gpytorch.kernels.Kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    noise_fix: bool = False
    opt_algo: torch.optim.Optimizer = torch.optim.Adam
    opt_algo_kwargs: dict = field(default_factory=dict)
    verbose_training: bool = False
    training_iter: int = 50
    prediction_strategy: Any = None


class Sogpr(TorchGPRegressor):
    def __init__(
        self,
        regressor=ExactGPModel,
        parameter: Sogpr_Parameters = Sogpr_Parameters(),
        train_data=None,
        design=None,
    ):

        super().__init__(
            regressor=regressor,
            parameter=parameter,
            train_data=train_data,
            design=design,
        )


class MultifidelityGPModel(GPyTorch_ExactGP):

    def __init__(self, train_x, train_y, likelihood):
        _imports.check()
        super(MultifidelityGPModel, self).__init__(train_x, train_y, likelihood)

    def __call__(self, *args, **kwargs):
        train_inputs = list(self.train_inputs) if self.train_inputs is not None else []
        # inputs = [i.unsqueeze(-1) if i.ndimension() == 1 else i for i in args]
        inputs = args[0]

        # Training mode: optimizing
        if self.training:
            if self.train_inputs is None:
                raise RuntimeError(
                    "train_inputs, train_targets cannot be None in training mode. "
                    "Call .eval() for prior predictions, or call .set_train_data() to add training data."
                )
            if gpytorch.settings.debug.on():
                if not all(torch.equal(train_input, input) for train_input, input in zip(train_inputs, inputs)):
                    raise RuntimeError("You must train on the training inputs!")
            res = super().__call__(*inputs, **kwargs)
            return res

        # Prior mode
        elif gpytorch.settings.prior_mode.on() or self.train_inputs is None or self.train_targets is None:
            full_inputs = args
            full_output = super(gpytorch.models.ExactGP, self).__call__(*full_inputs, **kwargs)
            if gpytorch.settings.debug().on():
                if not isinstance(full_output, gpytorch.distributions.MultivariateNormal):
                    raise RuntimeError("ExactGP.forward must return a MultivariateNormal")
            return full_output

        # Posterior mode
        else:
            if gpytorch.settings.debug.on():
                if all(torch.equal(train_input, input) for train_input, input in zip(train_inputs, inputs)):
                    warnings.warn(
                        "The input matches the stored training data. Did you forget to call model.train()?",
                        gpytorch.GPInputWarning,
                    )

            # Compute full output
            full_output = super(gpytorch.models.ExactGP, self).__call__(train_inputs, inputs, **kwargs)

            if gpytorch.settings.debug().on():
                if not isinstance(full_output, gpytorch.distributions.MultivariateNormal):
                    raise RuntimeError("ExactGP.forward must return a MultivariateNormal")
            full_mean, full_covar = full_output.loc, full_output.lazy_covariance_matrix

            # Determine the shape of the joint distribution
            batch_shape = full_output.batch_shape
            joint_shape = full_output.event_shape
            tasks_shape = joint_shape[1:]  # For multitask learning
            test_shape = torch.Size([joint_shape[0] - self.prediction_strategy.train_shape[0], *tasks_shape])

            # Make the prediction
            with gpytorch.settings.cg_tolerance(gpytorch.settings.eval_cg_tolerance.value()):
                predictive_mean, predictive_covar = self.prediction_strategy.exact_prediction(full_mean.float(), full_covar.float())

            # Reshape predictive mean to match the appropriate event shape
            predictive_mean = predictive_mean.view(*batch_shape, *test_shape).contiguous()
            return full_output.__class__(predictive_mean, predictive_covar)


class MultiFidelityPredictionStrategy(GPyTorch_PredictionStrategy):
    def __init__(self, train_inputs, train_prior_dist, train_labels, likelihood, root=None, inv_root=None):
        super().__init__(torch.cat(train_inputs), train_prior_dist, torch.cat(train_labels), likelihood, root, inv_root)
    
    @property
    @gpytorch.utils.memoize.cached(name="covar_cache")
    def covar_cache(self):
        train_train_covar = self.lik_train_train_covar
        train_train_covar_inv_root = linear_operator.to_dense(train_train_covar.root_inv_decomposition().root)
        return self._exact_predictive_covar_inv_quad_form_cache(train_train_covar_inv_root, self._last_test_train_covar).float()

    @property
    @gpytorch.utils.memoize.cached(name="mean_cache")
    def mean_cache(self):
        mvn = self.likelihood(self.train_prior_dist, self.train_inputs)
        train_mean, train_train_covar = mvn.loc, mvn.lazy_covariance_matrix

        train_labels_offset = (self.train_labels - train_mean).unsqueeze(-1)
        mean_cache = train_train_covar.evaluate_kernel().solve(train_labels_offset).squeeze(-1)

        if gpytorch.settings.detach_test_caches.on():
            mean_cache = mean_cache.detach()

        if mean_cache.grad_fn is not None:
            wrapper = functools.partial(gpytorch.utils.memoize.clear_cache_hook, self)
            functools.update_wrapper(wrapper, gpytorch.utils.memoize.clear_cache_hook)
            mean_cache.grad_fn.register_hook(wrapper)

        return mean_cache.float()

@dataclass
class Cokgj_Parameters:
    """(Pre-initialized) hyperparameters for single-fidelity Gaussian process regression in pytorch"""

    likelihood: gpytorch.likelihoods.Likelihood = gpytorch.likelihoods.GaussianLikelihood()
    mean: gpytorch.means.Mean or List[gpytorch.means.Mean] = gpytorch.means.ZeroMean()
    kernel: gpytorch.kernels.Kernel or List[gpytorch.kernels.Kernel] = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    noise_fix: bool = False
    opt_algo: torch.optim.Optimizer = torch.optim.Adam
    opt_algo_kwargs: dict = field(default_factory=dict)
    verbose_training: bool = False
    training_iter: int = 50
    prediction_strategy: MultiFidelityPredictionStrategy = MultiFidelityPredictionStrategy


class CokgjModel(MultifidelityGPModel):
    
    num_outputs = 1

    def __init__(self, train_x, train_y, likelihood, mean_module, covar_module, rho=[1.87], noise=[0., 0.]):
        super(CokgjModel, self).__init__(train_x, train_y, likelihood)
        self.train_x = train_x
        self.train_y = train_y
        self.likelihood = likelihood
        self.mean_module_list = mean_module
        self.covar_module_list = covar_module
        # self.rho = rho
        self.noise = noise # noiseless Forrester problem
        self.rho = torch.nn.Parameter(torch.zeros(1)) #TODO: add parameter bounds (rho)
        # self.noise = torch.nn.Parameter(torch.ones(2)) #TODO: add parameter bounds (noises)
        # self.prediction_strategy = self._set_prediction_strategy()

    def forward(self, x, x2=None):
        # assume: x is a list; x[0] contains design points of the lowest fidelity
        # and x[-1] contains design points of the highest fidelity

        if x2 is None:
            mean_x = self._construct_mean_module(x)
            covar_x = self._construct_covar_module(x)

        else: # construct "Kronecker" matrix
            mean_x1 = self._construct_mean_module(x)
            mean_x2 = self._construct_mean_module(x2)
            mean_x = torch.cat([mean_x1, mean_x2])

            covar_x11 = self._construct_covar_module(x)
            covar_x12 = self._construct_covar_module(x, x2)
            covar_x22 = self._construct_covar_module(x2, x2)

            covar_x1 = linear_operator.operators.CatLinearOperator(
                covar_x11, covar_x12, dim=1,
            )

            covar_x2 = linear_operator.operators.CatLinearOperator(
                covar_x12.T.float(), covar_x22, dim=1,
            )
            
            covar_x = linear_operator.operators.CatLinearOperator(
                covar_x1, covar_x2, dim=0,
            )
            pass

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def _construct_mean_module(self, x, x2=None):
        mean_x_list = []
        for mean_module, x_fid in zip(self.mean_module_list, x):
            mean_x_fid = mean_module(x_fid)
            mean_x_list.append(mean_x_fid)
            pass
        mean_x = torch.cat(mean_x_list)
        return mean_x

    def _construct_covar_module(self, x, x2=None):

        if x2 is None:
            x2 = x
        
        cov_block_matrix = None
        for i, x_i in enumerate(x):
            cov_block_matrix_row = None
            for j, x_j in enumerate(x2):
                m = min(i, j)
                cov_matrix_i_j_total = 0
                for k in range(m + 1):
                    rho_prod_i = math.prod(self.rho[k:i])
                    rho_prod_j = math.prod(self.rho[k:j])
                    cov_i_j = self.covar_module_list[k](x_i, x_j)
                    if i == j and x_i.shape == x_j.shape:
                        cov_i_j += self.noise[k] ** 2 * linear_operator.operators.IdentityLinearOperator(diag_shape=x_i.shape[0])
                    cov_matrix_i_j = rho_prod_i * rho_prod_j * cov_i_j
                    cov_matrix_i_j_total += cov_matrix_i_j

                if cov_block_matrix_row is None:
                    cov_block_matrix_row = cov_matrix_i_j_total
                else:
                    cov_block_matrix_row = linear_operator.operators.CatLinearOperator(
                        cov_block_matrix_row, cov_matrix_i_j_total, dim=1
                    )
            
            if cov_block_matrix is None:
                cov_block_matrix = cov_block_matrix_row
            else:
                cov_block_matrix = linear_operator.operators.CatLinearOperator(
                    cov_block_matrix, cov_block_matrix_row, dim=0,
                )
                
        return cov_block_matrix 


class Cokgj(TorchGPRegressor):
    def __init__(
        self,
        # regressor=CoKrigingGP,
        regressor=CokgjModel,
        parameter=Cokgj_Parameters(),
        train_data=None,
        design=None,
    ):
        super().__init__(
            parameter=parameter,
            regressor=regressor,
            train_data=train_data,
            design=design,
        )
    
    def data_to_x_y(self):
        train_x_list = []
        train_y_list = []
        for train_data_fid in self.train_data:
            train_x = torch.tensor(train_data_fid.get_input_data().values)
            train_y = torch.tensor(train_data_fid.get_output_data().values).flatten()

            train_x_list.append(train_x)
            train_y_list.append(train_y)
        return train_x_list, train_y_list


class MultitaskGPModel(MultifidelityGPModel):
    def __init__(self, train_x, train_y, likelihood, mean_module, covar_module):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.train_x = train_x
        self.train_y = train_y
        self.mean_module = mean_module # gpytorch.means.ConstantMean()
        self.covar_module = covar_module # gpytorch.kernels.RBFKernel()

        # We learn an IndexKernel for 2 tasks
        # (so we'll actually learn 2x2=4 tasks with correlations)
        self.task_covar_module = gpytorch.kernels.IndexKernel(num_tasks=2, rank=1)

    # def forward(self, x, i):
    def forward(self, x, x2=None):
        # assume: x is a list; x[0] contains design points of the lowest fidelity
        # and x[-1] contains design points of the highest fidelity

        if x2 is not None:
            for k, (x_k, x2_k) in enumerate(zip(x, x2)):
                x[k] = torch.cat([x_k, x2_k])

        i = []
        for k, x_k in enumerate(x):
            i.append(k * torch.ones(x_k.shape))

        i = torch.cat(i)
        x = torch.cat(x)

        mean_x = self.mean_module(x)

        # Get input-input covariance
        covar_x = self.covar_module(x)
        # Get task-task covariance
        covar_i = self.task_covar_module(i)
        # Multiply the two together to get the covariance we want
        covar = covar_x.mul(covar_i)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar)
    

@dataclass
class Multitask_Parameters:
    """(Pre-initialized) hyperparameters for single-fidelity Gaussian process regression in pytorch"""

    likelihood: gpytorch.likelihoods.Likelihood = gpytorch.likelihoods.GaussianLikelihood()
    mean: gpytorch.means.Mean or List[gpytorch.means.Mean] = gpytorch.means.ZeroMean()
    kernel: gpytorch.kernels.Kernel or List[gpytorch.kernels.Kernel] = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    noise_fix: bool = False
    opt_algo: torch.optim.Optimizer = torch.optim.Adam
    opt_algo_kwargs: dict = field(default_factory=dict)
    verbose_training: bool = False
    training_iter: int = 50
    prediction_strategy: MultiFidelityPredictionStrategy = MultiFidelityPredictionStrategy
    # linear_truncated: bool = False
    # data_fidelity: int = -1

class MultitaskGPR(TorchGPRegressor):
    def __init__(
        self,
        # regressor=SingleTaskMultiFidelityGP,
        regressor=MultitaskGPModel,
        parameter=Multitask_Parameters(),
        mf_train_data=None,
        design=None,
        # noise_fix: bool = False
    ):
        super().__init__(
            train_data=mf_train_data,
            # linear_truncated=parameter.linear_truncated,
            # data_fidelity=parameter.data_fidelity,
            regressor=regressor,
            parameter=parameter,
            design=design,
            # noise_fix=noise_fix,
        )

        self.regressor = regressor

    def data_to_x_y(self):
        train_x_list = []
        train_y_list = []
        for train_data_fid in self.train_data:
            train_x = torch.tensor(train_data_fid.get_input_data().values)
            train_y = torch.tensor(train_data_fid.get_output_data().values).flatten()

            train_x_list.append(train_x)
            train_y_list.append(train_y)
        return train_x_list, train_y_list