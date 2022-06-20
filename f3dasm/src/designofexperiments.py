from dataclasses import dataclass, field
from typing import List

from f3dasm.src.space import (
    CategoricalSpace,
    ContinuousSpace,
    DiscreteSpace,
    SpaceInterface,
)


@dataclass
class DoE:
    """Design of experiments

    Args:
        space (list): list of parameters
    """

    space: List[SpaceInterface] = field(default_factory=list)

    def addSpace(self, space: SpaceInterface) -> None:
        """Add a new parameter to the searchspace

        Args:
            space (SpaceInterface): search space parameter to be added
        """
        self.space.append(space)
        return

    def getContinuousParameters(self) -> list[ContinuousSpace]:
        """Receive all the continuous parameters"""
        return [
            parameter
            for parameter in self.space
            if isinstance(parameter, ContinuousSpace)
        ]

    def getContinuousNames(self) -> list[str]:
        """Receive all the continuous parameter names"""
        return [
            parameter.name
            for parameter in self.space
            if isinstance(parameter, ContinuousSpace)
        ]

    def getDiscreteParameters(self) -> list[DiscreteSpace]:
        """Receive all the discrete parameters"""
        return [
            parameter
            for parameter in self.space
            if isinstance(parameter, DiscreteSpace)
        ]

    def getDiscreteNames(self) -> list[str]:
        """Receive all the continuous parameter names"""
        return [
            parameter.name
            for parameter in self.space
            if isinstance(parameter, DiscreteSpace)
        ]

    def getCategoricalParameters(self) -> list[CategoricalSpace]:
        """Receive all the categorical parameters"""
        return [
            parameter
            for parameter in self.space
            if isinstance(parameter, CategoricalSpace)
        ]

    def getCategoricalNames(self) -> list[str]:
        """Receive all the continuous parameter names"""
        return [
            parameter.name
            for parameter in self.space
            if isinstance(parameter, CategoricalSpace)
        ]
