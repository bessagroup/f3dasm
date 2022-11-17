from typing import Union, Optional

import torch
from botorch.acquisition.objective import PosteriorTransform
from botorch.acquisition.analytic import AnalyticAcquisitionFunction
from botorch.models.model import Model
from botorch.utils import t_batch_mode_transform
from torch import Tensor


class VFUpperConfidenceBound(AnalyticAcquisitionFunction):
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
            model: Model,
            beta: Union[float, Tensor],
            posterior_transform: Optional[PosteriorTransform] = None,
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

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate the Upper Confidence Bound on the candidate set X.

        Args:
            X: A `(b1 x ... bk) x 1 x d`-dim batched tensor of `d`-dim design points.

        Returns:
            A `(b1 x ... bk)`-dim tensor of Upper Confidence Bound values at the
            given design points `X`.
        """
        # self.beta = self.beta.to(X)
        X_high = X.clone()
        X_high[0, 0, -1] = 1
        posterior_high = self.model.posterior(
            X=X_high, posterior_transform=self.posterior_transform
        )
        mean_high = posterior_high.mean
        view_shape = mean_high.shape[:-2] if mean_high.shape[-2] == 1 else mean_high.shape[:-1]
        mean_high = mean_high.view(view_shape)
        variance_high = posterior_high.variance.view(view_shape)

        if X_high[0, 0, -1] != X[0, 0, -1]:
            posterior_low = self.model.posterior(
                X=X, posterior_transform=self.posterior_transform
            )

            variance_low = posterior_low.variance.view(view_shape)

            sigma = self.cr * variance_low.sqrt()

        else:

            sigma = variance_high.sqrt()

        omega_1, omega_2 = self.weights()

        if self.maximize:
            res = omega_1 * mean_high + omega_2 * sigma
        else:
            res = -omega_1 * mean_high + omega_2 * sigma

        return res

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