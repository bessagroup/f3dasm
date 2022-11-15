from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional

from ..base.design import DesignSpace
from ..base.data import Data



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
    train_data: Data
    design: DesignSpace
    hyperparameters: Optional[Mapping[str, Any]] = field(default_factory=dict)

    def set_train_data(
        self,
        train_data: Data,
    ) -> None:
        self.train_data = train_data

    def set_regressor(self) -> None:
        pass

    def train(self) -> Surrogate:
        pass
