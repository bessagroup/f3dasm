#                                                                       Modules
# =============================================================================

# Standard
from typing import Union, Optional

# Local
from .._imports import try_import

# Third-party extension
with try_import('machinelearning') as _imports:
    import torch
    import botorch
    from botorch.acquisition.analytic import AnalyticAcquisitionFunction as BOTorch_AnalyticAcquisitionFunction
    import gpytorch

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Leo Guo (L.L.Guo@tudelft.nl)'
__credits__ = ['Leo Guo']
__status__ = 'Alpha'
# =============================================================================
#
# =============================================================================

if not _imports.is_successful():
    BOTorch_AnalyticAcquisitionFunction = object # NOQA


class Acquisition: #TODO: implement class
    pass


class UpperConfidenceBound(BOTorch_AnalyticAcquisitionFunction):
    r"""Single-outcome Upper Confidence Bound (UCB).

    Analytic upper confidence bound that comprises of the posterior mean plus an
    additional term: the posterior standard deviation weighted by a trade-off
    parameter, `beta`. Only supports the case of `q=1` (i.e. greedy, non-batch
    selection of design points). The model must be single-outcome.

    `UCB(x) = mu(x) + sqrt(beta) * sigma(x)`, where `mu` and `sigma` are the
    posterior mean and standard deviation, respectively.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> UCB = UpperConfidenceBound(model, beta=0.2)
        >>> ucb = UCB(test_X)
    """

    def __init__(
        self,
        model: botorch.models.model.Model,
        beta: Union[float, torch.Tensor],
        posterior_transform: Optional[botorch.acquisition.objective.PosteriorTransform] = None,
        maximize: bool = True,
        **kwargs,
    ) -> None:
        r"""Single-outcome Upper Confidence Bound.

        Args:
            model: A fitted single-outcome GP model (must be in batch mode if
                candidate sets X will be)
            beta: Either a scalar or a one-dim tensor with `b` elements (batch mode)
                representing the trade-off parameter between mean and covariance
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, posterior_transform=posterior_transform, **kwargs)
        self.maximize = maximize
        if not torch.is_tensor(beta):
            beta = torch.tensor(beta)
        self.register_buffer("beta", beta)

    @botorch.utils.t_batch_mode_transform(expected_q=1)
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        r"""Evaluate the Upper Confidence Bound on the candidate set X.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.

        Returns:
            A `(b1 x ... bk)`-dim tensor of Upper Confidence Bound values at the
            given design points `X`.
        """

        # Get into evaluation (predictive posterior) mode
        self.model.eval()
        self.model.likelihood.eval()

        # # Test points are regularly spaced along [0,1]
        # # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.model.likelihood(self.model(torch.tensor(X[None, :])))
        
        mean = observed_pred.mean
        variance = observed_pred.variance

        delta = (self.beta.expand_as(mean) * variance).sqrt()

        if self.maximize:
            res = (mean + delta).flatten()#.reshape(X.shape)
        else:
            res = (-mean + delta).flatten()#.reshape(X.shape)
        
        return res


class ExpectedImprovement(BOTorch_AnalyticAcquisitionFunction):
    r"""Single-outcome Expected Improvement (analytic).

    Computes classic Expected Improvement over the current best observed value,
    using the analytic formula for a Normal posterior distribution. Unlike the
    MC-based acquisition functions, this relies on the posterior at single test
    point being Gaussian (and require the posterior to implement `mean` and
    `variance` properties). Only supports the case of `q=1`. The model must be
    single-outcome.

    `EI(x) = E(max(y - best_f, 0)), y ~ f(x)`

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> EI = ExpectedImprovement(model, best_f=0.2)
        >>> ei = EI(test_X)
    """

    def __init__(
        self,
        model: botorch.models.model.Model,
        best_f: Union[float, torch.Tensor],
        posterior_transform: Optional[botorch.acquisition.objective.PosteriorTransform] = None,
        maximize: bool = True,
        **kwargs,
    ) -> None:
        r"""Single-outcome Expected Improvement (analytic).

        Args:
            model: A fitted single-outcome model.
            best_f: Either a scalar or a `b`-dim Tensor (batch mode) representing
                the best function value observed so far (assumed noiseless).
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, posterior_transform=posterior_transform, **kwargs)
        self.maximize = maximize
        if not torch.is_tensor(best_f):
            best_f = torch.tensor(best_f)
        self.register_buffer("best_f", best_f)

    @botorch.utils.t_batch_mode_transform(expected_q=1, assert_output_shape=False)
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        r"""Evaluate Expected Improvement on the candidate set X.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.
                Expected Improvement is computed for each point individually,
                i.e., what is considered are the marginal posteriors, not the
                joint.

        Returns:
            A `(b1 x ... bk)`-dim tensor of Expected Improvement values at the
            given design points `X`.
        """

        # Get into evaluation (predictive posterior) mode
        self.model.eval()
        self.model.likelihood.eval()

        # # Test points are regularly spaced along [0,1]
        # # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.model.likelihood(self.model(torch.tensor(X[None, :])))

        mean = observed_pred.mean
        sigma = observed_pred.variance.abs().sqrt()

        u = (mean - self.best_f.expand_as(mean)) / sigma
        if not self.maximize:
            u = -u
        normal = torch.distributions.Normal(torch.zeros_like(u), torch.ones_like(u))
        ucdf = normal.cdf(u)
        updf = torch.exp(normal.log_prob(u))
        ei = sigma * (updf + u * ucdf)
        return ei


class VFUpperConfidenceBound(BOTorch_AnalyticAcquisitionFunction):
    r"""Single-outcome Upper Confidence Bound (UCB).

    Analytic upper confidence bound that comprises the posterior mean plus an
    additional term: the posterior standard deviation weighted by a trade-off
    parameter, `beta`. Only supports the case of `q=1` (i.e. greedy, non-batch
    selection of design points). The model must be single-outcome.

    `UCB(x) = mu(x) + sqrt(beta) * sigma(x)`, where `mu` and `sigma` are the
    posterior mean and standard deviation, respectively.

    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> UCB = UpperConfidenceBound(model, beta=0.2)
        >>> ucb = UCB(test_X)
    """

    def __init__(
            self,
            model: botorch.models.model.Model,
            beta: Union[float, torch.Tensor],
            posterior_transform: Optional[botorch.acquisition.objective.PosteriorTransform] = None,
            maximize: bool = True,
            cr: float = 10,
            mean=None,
            var=None,
            **kwargs,
    ) -> None:
        r"""Single-outcome Upper Confidence Bound.

        Args:
            model: A fitted single-outcome GP model (must be in batch mode if
                candidate sets X will be)
            beta: Either a scalar or a one-dim tensor with `b` elements (batch mode)
                representing the trade-off parameter between mean and covariance
            posterior_transform: A PosteriorTransform. If using a multi-output model,
                a PosteriorTransform that transforms the multi-output posterior into a
                single-output posterior is required.
            maximize: If True, consider the problem a maximization problem.
        """
        super().__init__(model=model, posterior_transform=posterior_transform, **kwargs)
        self.maximize = maximize
        if not torch.is_tensor(beta):
            beta = torch.tensor(beta)
        self.register_buffer("beta", beta)
        self.cr = cr
        self.mean = mean
        self.var = var

    @botorch.utils.t_batch_mode_transform(expected_q=1)
    def forward(self, X: torch.Tensor, fid: int) -> torch.Tensor:
        r"""Evaluate the Upper Confidence Bound on the candidate set X.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.

        Returns:
            A `(b1 x ... bk)`-dim tensor of Upper Confidence Bound values at the
            given design points `X`.
        """
        # self.beta = self.beta.to(X)

        self.model.eval()
        self.model.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred_high = self.model([torch.empty(0, X.shape[-1]), X[0]])
            observed_pred_low = self.model([X[0], torch.empty(0, X.shape[-1])])

        mean_high = observed_pred_high.mean
        variance_low = observed_pred_low.variance
        variance_high = observed_pred_high.variance

        if fid == 0:
            sigma = torch.sqrt(variance_low)
            CR = torch.tensor(self.cr)
        else:
            sigma = torch.sqrt(variance_low + variance_high)
            CR = torch.tensor(1.)

        omega_1, omega_2 = self.weights()

        if self.maximize:
            res = omega_1 * mean_high + omega_2 * sigma * CR
        else:
            res = -omega_1 * mean_high + omega_2 * sigma * CR

        return res.flatten()

    def weights(self):
        sample_mean_lf = torch.mean(torch.tensor(self.mean[0]))
        sample_std_lf = torch.std(torch.tensor(self.mean[0]))
        sample_mean_hf = torch.mean(torch.tensor(self.mean[1]))
        sample_std_hf = torch.std(torch.tensor(self.mean[1]))

        a1 = sample_mean_hf / sample_std_hf
        a2 = sample_mean_lf / sample_std_lf

        omega_1 = a1 / (a1 + a2)
        omega_2 = a2 / (a1 + a2)

        return omega_1, omega_2