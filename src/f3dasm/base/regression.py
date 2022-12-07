#                                                                       Modules
# =============================================================================

# Standard
from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional

# Locals
from ..base.data import Data
from ..base.design import DesignSpace

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
        test_input_data: Data,
    ) -> Data or List[Data]:
        pass

    def save_model(self):
        pass


@dataclass
class Regressor:
    train_input_data: Data
    train_output_data: Data
    design: DesignSpace
    hyperparameters: Optional[Mapping[str, Any]] = field(default_factory=dict)

    def set_train_data(
        self,
        train_input_data: Data,
        train_output_data: Data,
    ) -> None:
        self.train_input_data = train_input_data
        self.train_output_data = train_output_data

    def set_regressor(self) -> None:
        pass

    def train(self) -> Surrogate:
        pass
