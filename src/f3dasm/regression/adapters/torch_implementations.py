from dataclasses import dataclass
from typing import Any, List

import torch
import gpytorch
from botorch import fit_gpytorch_model
from botorch.models.transforms import Normalize, Standardize
from gpytorch import ExactMarginalLogLikelihood

from f3dasm import Data, ContinuousParameter
from f3dasm.base.regression import Regressor, Surrogate


# from f3dasm.regression.gpr import Mtask

# def _continuousInputSpace_to_tensor(input_space: List[ContinuousParameter]) -> torch.tensor:
#     input_space_list = []
#     for continuousParameter in input_space:
#         input_space_list.append([continuousParameter.lower_bound, continuousParameter.upper_bound])
#     input_space_tensor = torch.tensor(input_space_list).T
#     return input_space_tensor


@dataclass
class TorchGPSurrogate(Surrogate):
    likelihood: Any = None
    regressor_class: Any = None

    def _data_processor(
        self,
        test_input_data: Data
    ): # processes data into the appropriate input array given the regressor

        input_df = test_input_data.data['input']

        if self.regressor_class.__name__ == 'Mtask':
            input_array = input_df.drop('fid', axis=1).to_numpy()[:, None]
        else:
            input_array = input_df.to_numpy()

        return input_array

    def predict(
            self,
            test_input_data: Data,
    ) -> List[Data]:

        input_array = self._data_processor(test_input_data=test_input_data)

        torch_posterior = self.model.posterior(
            torch.from_numpy(input_array)  # .to(**tkwargs)
        )

        if self.regressor_class.__name__ == 'Mtask':
            test_y_list = torch_posterior.mean.cpu().detach().numpy().squeeze(axis=1)[:, 1][:, None]
            test_y_var_list = torch_posterior.variance.detach().numpy().squeeze(axis=1)[:, 1][:, None]
        else:
            test_y_list = torch_posterior.mean.cpu().detach().numpy()
            test_y_var_list = torch_posterior.mvn.covariance_matrix.diag().cpu().detach().numpy()[:, None]

        return [test_y_list, test_y_var_list]

    def save_model(self):
        pass


# @dataclass
class TorchGPRegressor(Regressor):

    def __init__(
            self,
            regressor=None,
            train_data: Data = None,
            design=None,
            noise_fix: bool = False,
            **kwargs
    ):
        super().__init__(
            train_data=train_data,
            design=design,
        )
        self.regressor = regressor
        self.noise_fix = noise_fix
        self.kwargs = kwargs

    def train(self) -> Surrogate:
        surrogate = TorchGPSurrogate(
            model=self.regressor(
                train_X=torch.tensor(self.train_data.get_input_data().values),
                train_Y=torch.tensor(self.train_data.get_output_data().values),
                # input_transform=Normalize(
                #     d=len(self.design.input_space),
                #     # bounds=_continuousInputSpace_to_tensor(self.design.input_space),
                #     # bounds=None,
                # ),
                # outcome_transform=Standardize(m=1),
                **self.kwargs,
            ),
            regressor_class=self.__class__
        )

        if self.noise_fix:
            # surrogate.model.likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Interval(1e-4, 2e-4))
            surrogate.model.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=torch.tensor([1e-4]))

        mll = ExactMarginalLogLikelihood(surrogate.model.likelihood, surrogate.model)
        fit_gpytorch_model(mll, max_retries=10)

        return surrogate
        