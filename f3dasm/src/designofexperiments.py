from dataclasses import dataclass, field
from typing import List

import pandas as pd


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
        input_space (List[SpaceInterface]): list of parameters, :class:`~f3dasm.src.space`, :class:`numpy.ndarray`
        output_space (List[SpaceInterface]): list of parameters, :class:`~f3dasm.src.space`, :class:`numpy.ndarray`
    """

    input_space: List[SpaceInterface] = field(default_factory=list)
    output_space: List[SpaceInterface] = field(default_factory=list)

    def get_empty_dataframe(self) -> pd.DataFrame:
        # input columns
        df_input = pd.DataFrame(
            columns=[("input", s.name) for s in self.input_space]
        ).astype(self.cast_types_dataframe(self.input_space, label="input"))

        # output columns
        df_output = pd.DataFrame(
            columns=[("output", s.name) for s in self.output_space]
        ).astype(self.cast_types_dataframe(self.output_space, label="output"))

        return pd.concat([df_input, df_output])

    def cast_types_dataframe(self, space: List[SpaceInterface], label: str) -> dict:
        # Make a dictionary that provides the datatype of each parameter
        return {(label, parameter.name): parameter.type for parameter in space}

    def add_input_space(self, space: SpaceInterface) -> None:
        """Add a new parameter to the searchspace

        Args:
            space (SpaceInterface): search space parameter to be added
        """
        self.input_space.append(space)
        return

    def add_output_space(self, space: SpaceInterface) -> None:
        """Add a new parameter to the searchspace

        Args:
            space (SpaceInterface): search space parameter to be added
        """
        self.output_space.append(space)
        return

    def get_input_space(self) -> List[SpaceInterface]:
        return self.input_space

    def get_output_space(self) -> List[SpaceInterface]:
        return self.output_space

    def all_input_continuous(self) -> bool:
        """Check if all input parameters are continuous"""
        return all(
            isinstance(parameter, ContinuousSpace) for parameter in self.input_space
        )

    def getNumberOfInputParameters(self) -> int:
        """Obtain the number of input parameters

        Returns:
            int: number of input parameters
        """
        return len(self.input_space)

    def getNumberOfOutputParameters(self) -> int:
        """Obtain the number of input parameters

        Returns:
            int: number of output parameters
        """
        return len(self.output_space)

    def getContinuousParameters(self) -> List[ContinuousSpace]:
        """Obtain all the continuous parameters

        Returns:
            List[ContinuousSpace]: space of continuous parameters
        """
        return [
            parameter
            for parameter in self.input_space
            if isinstance(parameter, ContinuousSpace)
        ]

    def getContinuousNames(self) -> List[str]:
        """Receive all the continuous parameter names"""
        return [
            parameter.name
            for parameter in self.input_space
            if isinstance(parameter, ContinuousSpace)
        ]

    def getDiscreteParameters(self) -> List[DiscreteSpace]:
        """Obtain all the discrete parameters

        Returns:
            List[DiscreteSpace]: space of discrete parameters
        """
        return [
            parameter
            for parameter in self.input_space
            if isinstance(parameter, DiscreteSpace)
        ]

    def getDiscreteNames(self) -> List[str]:
        """Receive the names of all the discrete parameters

        Returns:
            List[str]: list of names
        """
        return [
            parameter.name
            for parameter in self.input_space
            if isinstance(parameter, DiscreteSpace)
        ]

    def getCategoricalParameters(self) -> List[CategoricalSpace]:
        """Obtain all the categorical parameters

        Returns:
            List[CategoricalSpace]: space of categorical parameters
        """
        return [
            parameter
            for parameter in self.input_space
            if isinstance(parameter, CategoricalSpace)
        ]

    def getCategoricalNames(self) -> List[str]:
        """Receive the names of all the categorical parameters

        Returns:
            List[str]: list of names
        """
        return [
            parameter.name
            for parameter in self.input_space
            if isinstance(parameter, CategoricalSpace)
        ]
