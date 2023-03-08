from dataclasses import dataclass
from typing import Any, List

import torch
import gpytorch
# from botorch import fit_gpytorch_model
# from botorch.models.transforms import Normalize, Standardize
# from gpytorch import ExactMarginalLogLikelihood

# from .. import Data, ContinuousParameter
from ...design import ExperimentData
from ...base.regression import Regressor, Surrogate

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Leo Guo (L.L.Guo@tudelft.nl)'
__credits__ = ['Leo Guo']
__status__ = 'Alpha'
# =============================================================================
#
# =============================================================================

@dataclass
class TorchGPSurrogate(Surrogate):
    likelihood: Any = None
    regressor_class: Any = None
    parameter: Any = None

    def _data_processor(
        self,
        test_input_data: ExperimentData
    ): # processes data into the appropriate input array given the regressor

        input_df = test_input_data.data['input']

        if self.regressor_class.__name__ == 'Mtask':
            input_array = input_df.drop('fid', axis=1).to_numpy()[:, None]
        else:
            input_array = input_df.to_numpy()

        return input_array

    def predict(
        self,
        test_x
    ):
        # Get into evaluation (predictive posterior) mode
        self.model.eval()
        self.model.likelihood.eval()

        if self.parameter.prediction_strategy is not None:
            self.model.prediction_strategy = self.parameter.prediction_strategy(
                train_inputs=self.model.train_x,
                train_prior_dist=self.model.forward(self.model.train_x),
                train_labels=self.model.train_y,
                likelihood=self.parameter.likelihood,
            )

        # # Make predictions by feeding model through likelihood
        with torch.no_grad(), gpytorch.settings.fast_pred_var():            
            observed_pred = self.model.likelihood(self.model(test_x))
    
        return observed_pred

    def optimal_hyperparameter(self):
        pass

    def save_model(self):
        pass

    def plot_gpr(
        self,
        # test_input_data: Data,
        test_x,
        scaler: StandardScaler,
        exact_y,
        observed_pred: gpytorch.distributions.multivariate_normal.MultivariateNormal,
        train_x=None,
        train_y=None,
        ):

        # input_array = self._data_processor(test_input_data=test_input_data)

        # Get into evaluation (predictive posterior) mode
        self.model.eval()
        self.model.likelihood.eval()

        with torch.no_grad():
            # Initialize plot
            f, ax = plt.subplots(1, 1)#, figsize=(4, 3))

            # Get upper and lower confidence bounds
            lower, upper = [torch.tensor(scaler.inverse_transform(confbound.numpy()[:, None]).flatten()) for confbound in observed_pred.confidence_region()]
            # lower, upper = observed_pred.confidence_region()

            # Plot ground truth as black dashed line
            ax.plot(test_x, exact_y, 'k--')
            # Plot predictive means as blue line
            ax.plot(test_x, scaler.inverse_transform(observed_pred.mean.numpy()[:, None]).flatten(), 'b')
            # Shade between the lower and upper confidence bounds
            ax.fill_between(test_x, lower.numpy(), upper.numpy(), alpha=0.5)
            # ax.set_ylim([-3, 3])
            if not (train_x is None or train_y is None):
                # Plot training data as black stars
                ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
                ax.legend(['Ground truth', 'Mean', 'Confidence', 'Observed Data'])
            else:
                ax.legend(['Ground truth', 'Mean', 'Confidence'])
            plt.tight_layout()

    def plot_mll(self, train_x, train_y_scaled):

        #TODO: add possibility to plot other parameters together

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.model.likelihood, self.model)

        opt_pars = [
            torch.tensor(self.model.likelihood.noise.item()),
            torch.tensor(self.model.covar_module.outputscale.item()),
            torch.tensor(self.model.covar_module.base_kernel.lengthscale.item())
            ]

        # print()
        # for i, (name, _) in enumerate(self.model.named_parameters()):
        #     print(name, opt_pars[i].item())

        self.model.train()
        self.model.likelihood.train()

        noise_level = torch.logspace(torch.log10(opt_pars[0]) - 0.1, torch.log10(opt_pars[0]) + 0.1, steps=30)
        amp_scale = torch.logspace(torch.log10(opt_pars[1]) - 0.5, torch.log10(opt_pars[1]) + 0.5, steps=30)
        length_scale = torch.logspace(torch.log10(opt_pars[2]) - 0.5, torch.log10(opt_pars[2]) + 0.5, steps=30)

        # length_scale_grid, noise_scale_grid = torch.meshgrid(length_scale, noise_level)
        length_scale_grid, amp_scale_grid = torch.meshgrid(length_scale, amp_scale)

        mll_plot_list = []
        # for scale, noise in zip(length_scale_grid.ravel(), noise_scale_grid.ravel()):
        for scale, amp in zip(length_scale_grid.ravel(), amp_scale_grid.ravel()):
            self.model.covar_module.base_kernel.lengthscale = scale
            # model.likelihood.noise = noise
            self.model.covar_module.outputscale = amp
            mll_plot_list.append(mll(self.model(train_x), train_y_scaled))

        mll_plot = torch.tensor(mll_plot_list).reshape(length_scale_grid.shape)
        mll_plot -= max(0, np.abs(np.amax(mll_plot.numpy()))) + 1e-3

        plot_min = [z for z in zip(length_scale_grid.ravel(), amp_scale_grid.ravel())][np.argmax(mll_plot)]
        
        from matplotlib.colors import LogNorm
        vmin, vmax = (-mll_plot).min(), (-mll_plot).max()
        level = np.logspace(np.log10(vmin), np.log10(vmax), num=50)

        plt.figure()
        plt.contourf(
            length_scale_grid.numpy(),
            # noise_scale_grid.numpy(),
            amp_scale_grid.numpy(),
            -mll_plot.numpy(),
            # levels=250,
            levels=level,
            norm=LogNorm(vmin=vmin, vmax=vmax),
        )
        plt.colorbar()
        cs = plt.contour(
            length_scale_grid.numpy(),
            # noise_scale_grid.numpy(),
            amp_scale_grid.numpy(),
            -mll_plot.numpy(),
            # levels=250,
            levels=level,
            norm=LogNorm(vmin=vmin, vmax=vmax),
            colors='black',
            linewidths=.3,
        )
        # plt.clabel(cs, cs.levels[::5], inline=1, fontsize=10)
        plt.plot(opt_pars[2], opt_pars[1], '.', color='white', label='Located optimum')
        plt.plot(plot_min[0], plot_min[1], 'r*', label='Optimum in range')
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Length-scale")
        # plt.ylabel("Noise-level")
        plt.ylabel("Output-scale")
        plt.title("Negative marginal log likelihood")
        plt.legend()
        plt.tight_layout()

    def gp_metrics(self, scaler, observed_pred, exact_y):
        pred_mean = scaler.inverse_transform(observed_pred.mean.numpy()[:, None]).flatten()

        l1_norm = np.linalg.norm(exact_y - pred_mean, ord=1)
        l2_norm = np.linalg.norm(exact_y - pred_mean, ord=2)
        max_norm = np.linalg.norm(exact_y - pred_mean, ord=np.inf)

        # print()
        # print('NORMS', l1_norm, l2_norm, max_norm)

        MAE = l1_norm / len(exact_y)
        MSE = l2_norm ** 2 / len(exact_y)

        # print()
        # print('MEAN ERRORS', MAE, MSE, max_norm)

        sample_mean = np.mean(exact_y)
        sample_MAD = np.mean(np.abs(exact_y - sample_mean))
        sample_var = np.var(exact_y)
        sample_max_diff = np.linalg.norm(exact_y - sample_mean, ord=np.inf)

        # print()
        # print('SAMPLE STATS', sample_MAD, sample_var, sample_max_diff)

        FMU1 = MAE / sample_MAD
        FMU2 = MSE / sample_var
        FMUinf = max_norm / sample_max_diff

        Rsq1, Rsq2, Rsqinf = 1 - FMU1, 1 - FMU2, 1 - FMUinf

        # print()
        # print('COEFFICIENTS OF DETERMINATION', Rsq1, Rsq2, Rsqinf)

        metrics_array = np.array(
            [
                [l1_norm, l2_norm, max_norm],
                [MAE, MSE, max_norm],
                [sample_MAD, sample_var, sample_max_diff],
                [Rsq1, Rsq2, Rsqinf],
            ]
        )

        metrics_df = pd.DataFrame(
            data=metrics_array, 
            columns=['1', '2', 'inf'],
            index=['p-distance', 'mean p-error', 'sample p-deviation', 'R^2_p']
            )
        
        return metrics_df


# @dataclass
class TorchGPRegressor(Regressor):

    def __init__(
            self,
            regressor=None,
            parameter=None,
            train_data: ExperimentData or List[ExperimentData] = None,
            design=None,
    ):
        super().__init__(
            train_data=train_data,
            design=design,
        )
        self.regressor = regressor
        self.parameter = parameter

        self.train_x, self.train_y = self.data_to_x_y()

        self.surrogate = TorchGPSurrogate(
            model=self.regressor(
                train_x=self.train_x,
                train_y=self.train_y,
                # input_transform=Normalize(
                #     d=len(self.design.input_space),
                #     # bounds=_continuousInputSpace_to_tensor(self.design.input_space),
                #     # bounds=None,
                # ),
                # outcome_transform=Standardize(m=1),
                likelihood=self.parameter.likelihood,
                covar_module=self.parameter.kernel,
                mean_module=self.parameter.mean,
                # **self.kwargs,
            ),
            regressor_class=self.__class__,
            parameter=self.parameter
        )

    def data_to_x_y(self):
        train_x = torch.tensor(self.train_data.get_input_data().values)
        train_y = torch.tensor(self.train_data.get_output_data().values).flatten()
        return train_x, train_y

    def train(
        self,
    ) -> Surrogate:

        if self.parameter.noise_fix:
            self.surrogate.model.likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Interval(1e-6, 1.1e-6))
            # surrogate.model.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=torch.tensor([1e-4]))

        # Get into training mode to find optimal model hyperparameters
        self.surrogate.model.train()
        self.surrogate.model.likelihood.train()

        # Use the adam optimizer
        optimizer = self.parameter.opt_algo(self.surrogate.model.parameters(), **self.parameter.opt_algo_kwargs)  # Includes GaussianLikelihood parameters
        # optimizer = self.parameter.opt_algo(self.surrogate.model.parameters(), lr=0.25)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.surrogate.model.likelihood, self.surrogate.model)

        for i in range(self.parameter.training_iter):
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = self.surrogate.model.forward(self.train_x)
            # Calculate loss and backprop gradients
            if type(self.train_y) is list:
                self.train_y = torch.cat(self.train_y)
            loss = -mll(output, self.train_y)
            loss.backward()
            if self.parameter.verbose_training:#i % 10 == 0:
                print()
                print('Iter %d/%d' % (i + 1, self.parameter.training_iter), end=' - ')
                print('loss', "%.3f" % loss.item(), end=' - ')
                # for k in range(len(list(self.surrogate.model.parameters()))):
                #     print(
                #         list(self.surrogate.model.named_parameters())[k][0],
                #         "%.3f" % list(self.surrogate.model.constraints())[k].transform(list(self.surrogate.model.parameters())[k]).flatten().item(),
                #         end=' - '
                #     )
            optimizer.step()

        return self.surrogate
        