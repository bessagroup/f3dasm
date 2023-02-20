#                                                                       Modules
# =============================================================================

# Standard
from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional

from ..design.design import DesignSpace
# Locals
from ..design.experimentdata import ExperimentData

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Leo Guo (L.Guo@tudelft.nl)'
__credits__ = ['Leo Guo']
__status__ = 'Alpha'
# =============================================================================
#
# =============================================================================


@dataclass
class Surrogate:
    model: Any

    def predict(
        self,
        test_input_data: ExperimentData,
    ) -> ExperimentData or List[ExperimentData]:
        pass

    def save_model(self):
        pass


@dataclass
class Regressor:
    train_input_data: ExperimentData
    train_output_data: ExperimentData
    design: DesignSpace
    hyperparameters: Optional[Mapping[str, Any]] = field(default_factory=dict)

    def set_train_data(
        self,
        train_input_data: ExperimentData,
        train_output_data: ExperimentData,
    ) -> None:
        self.train_input_data = train_input_data
        self.train_output_data = train_output_data

    def set_regressor(self) -> None:
        pass

    def train(self) -> Surrogate:
        pass
