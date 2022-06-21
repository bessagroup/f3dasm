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


    def getNumberOfParameters(self) -> int:
        return len(self.space)
        
    def getContinuousParameters(self) -> List[ContinuousSpace]:
        """Receive all the continuous parameters"""
        return [
            parameter
            for parameter in self.space
            if isinstance(parameter, ContinuousSpace)
        ]

    def getContinuousNames(self) -> List[str]:
        """Receive all the continuous parameter names"""
        return [
            parameter.name
            for parameter in self.space
            if isinstance(parameter, ContinuousSpace)
        ]

    def getDiscreteParameters(self) -> List[DiscreteSpace]:
        """Receive all the discrete parameters"""
        return [
            parameter
            for parameter in self.space
            if isinstance(parameter, DiscreteSpace)
        ]

    def getDiscreteNames(self) -> List[str]:
        """Receive all the continuous parameter names"""
        return [
            parameter.name
            for parameter in self.space
            if isinstance(parameter, DiscreteSpace)
        ]

    def getCategoricalParameters(self) -> List[CategoricalSpace]:
        """Receive all the categorical parameters"""
        return [
            parameter
            for parameter in self.space
            if isinstance(parameter, CategoricalSpace)
        ]

    def getCategoricalNames(self) -> List[str]:
        """Receive all the continuous parameter names"""
        return [
            parameter.name
            for parameter in self.space
            if isinstance(parameter, CategoricalSpace)
        ]
