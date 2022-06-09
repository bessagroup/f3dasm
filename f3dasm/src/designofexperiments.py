# %%
from dataclasses import dataclass, field
from typing import List

from pandas import Categorical

from f3dasm.src.space import (
    CategoricalSpace,
    ConstraintInterface,
    ContinuousSpace,
    DiscreteSpace,
    SpaceInterface,
)


@dataclass
class DoE:

    space: List[SpaceInterface] = field(default_factory=list)

    def addSpace(self, space: SpaceInterface) -> None:
        self.space.append(space)
        return

    def getContinuousParameters(self):
        return [
            parameter
            for parameter in self.space
            if isinstance(parameter, ContinuousSpace)
        ]

    def getDiscreteParameters(self):
        return [
            parameter
            for parameter in self.space
            if isinstance(parameter, DiscreteSpace)
        ]

    def getCategoricalParameters(self):
        return [
            parameter
            for parameter in self.space
            if isinstance(parameter, CategoricalSpace)
        ]
