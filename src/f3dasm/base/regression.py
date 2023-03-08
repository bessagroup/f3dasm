from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional

from ..design import ExperimentData, DesignSpace
# from ..data import Data

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Leo Guo (L.L.Guo@tudelft.nl)'
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
    train_data: ExperimentData or List[ExperimentData]
    design: DesignSpace or List[DesignSpace]
    hyperparameters: Optional[Mapping[str, Any]] = field(default_factory=dict)

    def set_train_data(
        self,
        train_data: ExperimentData,
    ) -> None:
        self.train_data = train_data

    def set_regressor(self) -> None:
        pass

    def train(self) -> Surrogate:
        pass
