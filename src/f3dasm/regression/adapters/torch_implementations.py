#                                                                       Modules
# =============================================================================

# Standard
from dataclasses import dataclass
from typing import Any, List

# Third-party
import torch
from botorch import fit_gpytorch_model
from botorch.models.transforms import Normalize, Standardize
from gpytorch import ExactMarginalLogLikelihood

# Locals
from ...base.data import Data
from ...base.design import ContinuousParameter
from ...base.regression import Regressor, Surrogate

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Leo Guo (L.Guo@tudelft.nl'
__credits__ = ['Leo Guo']
__status__ = 'Alpha'
# =============================================================================
#
# =============================================================================


def _continuousInputSpace_to_tensor(input_space: List[ContinuousParameter]) -> torch.tensor:
    input_space_list = []
    for continuousParameter in input_space:
        input_space_list.append(
            [continuousParameter.lower_bound, continuousParameter.upper_bound])
    input_space_tensor = torch.tensor(input_space_list).T
    return input_space_tensor


@dataclass
class TorchGPSurrogate(Surrogate):
    likelihood: Any = None

    def predict(
            self,
            test_input_data: Data,
    ) -> List[Data]:
        test_y_list = self.model.posterior(
            torch.from_numpy(test_input_data)  # .to(**tkwargs)
        ).mean.cpu().detach().numpy()

        test_y_var_list = self.model.posterior(
            torch.from_numpy(
                test_input_data
            )  # .to(**tkwargs)
        ).mvn.covariance_matrix.diag().cpu().detach().numpy()[:, None]
        return [test_y_list, test_y_var_list]

    def save_model(self):
        pass


# @dataclass
class TorchGPRegressor(Regressor):

    def __init__(
            self,
            regressor=None,
            train_input_data=None,
            train_output_data=None,
            design=None,
            **kwargs
    ):
        super().__init__(
            train_input_data=train_input_data,
            train_output_data=train_output_data,
            design=design,
        )
        self.regressor = regressor
        self.kwargs = kwargs

    def train(self) -> Surrogate:
        surrogate = TorchGPSurrogate(
            model=self.regressor(
                train_X=torch.tensor(self.train_input_data),
                train_Y=torch.tensor(self.train_output_data),
                input_transform=Normalize(
                    d=len(self.design.input_space),
                    bounds=_continuousInputSpace_to_tensor(
                        self.design.input_space),
                ),
                outcome_transform=Standardize(m=1),
                **self.kwargs,
            )
        )

        mll = ExactMarginalLogLikelihood(
            surrogate.model.likelihood, surrogate.model)
        fit_gpytorch_model(mll)

        return surrogate
