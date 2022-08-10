from dataclasses import dataclass, field
from typing import List, TypeVar

import pandas as pd


from ..base.space import (
    CategoricalParameter,
    ContinuousParameter,
    DiscreteParameter,
    ParameterInterface,
)


@dataclass
class DesignSpace:
    """Design of experiments

    Args:
        input_space (List[SpaceInterface]): list of parameters
        output_space (List[SpaceInterface]): list of parameters
    """

    input_space: List[ParameterInterface] = field(default_factory=list)
    output_space: List[ParameterInterface] = field(default_factory=list)

    def get_empty_dataframe(self) -> pd.DataFrame:
        # input columns
        df_input = pd.DataFrame(columns=self.get_input_names()).astype(
            self._cast_types_dataframe(self.input_space, label="input")
        )

        # output columns
        df_output = pd.DataFrame(columns=self.get_output_names()).astype(
            self._cast_types_dataframe(self.output_space, label="output")
        )

        return pd.concat([df_input, df_output])

    def _cast_types_dataframe(self, space: List[ParameterInterface], label: str) -> dict:
        # Make a dictionary that provides the datatype of each parameter
        return {(label, parameter.name): parameter.type for parameter in space}

    def add_input_space(self, space: ParameterInterface) -> None:
        """Add a new parameter to the searchspace

        Args:
            space (SpaceInterface): search space parameter to be added
        """
        self.input_space.append(space)
        return

    def add_output_space(self, space: ParameterInterface) -> None:
        """Add a new parameter to the searchspace

        Args:
            space (SpaceInterface): search space parameter to be added
        """
        self.output_space.append(space)
        return

    def get_input_space(self) -> List[ParameterInterface]:
        return self.input_space

    def get_output_space(self) -> List[ParameterInterface]:
        return self.output_space

    def get_output_names(self) -> List[str]:
        return [("output", s.name) for s in self.output_space]

    def get_input_names(self) -> List[str]:
        return [("input", s.name) for s in self.input_space]

    def _check_space_on_type(self, type: TypeVar, space: List[ParameterInterface]) -> bool:
        return all(isinstance(parameter, type) for parameter in space)

    def _all_input_continuous(self) -> bool:
        """Check if all input parameters are continuous"""
        return self._check_space_on_type(ContinuousParameter, self.input_space)

    def _all_output_continuous(self) -> bool:
        """Check if all output parameters are continuous"""
        return self._check_space_on_type(ContinuousParameter, self.output_space)

    def is_single_objective_continuous(self) -> bool:
        """Check if the output is single objective and continuous"""
        return (
            self._all_input_continuous()
            and self._all_output_continuous()
            and self.get_number_of_output_parameters() == 1
        )

    def get_number_of_input_parameters(self) -> int:
        """Obtain the number of input parameters

        Returns:
            int: number of input parameters
        """
        return len(self.input_space)

    def get_number_of_output_parameters(self) -> int:
        """Obtain the number of input parameters

        Returns:
            int: number of output parameters
        """
        return len(self.output_space)

    def get_continuous_parameters(self) -> List[ContinuousParameter]:
        """Obtain all the continuous parameters

        Returns:
            List[ContinuousSpace]: space of continuous parameters
        """
        return self._get_parameters(ContinuousParameter, self.input_space)

    def get_continuous_names(self) -> List[str]:
        """Receive all the continuous parameter names"""
        return self._get_names(ContinuousParameter, self.input_space)

    def get_discrete_parameters(self) -> List[DiscreteParameter]:
        """Obtain all the discrete parameters

        Returns:
            List[DiscreteSpace]: space of discrete parameters
        """
        return self._get_parameters(DiscreteParameter, self.input_space)

    def get_discrete_names(self) -> List[str]:
        """Receive the names of all the discrete parameters

        Returns:
            List[str]: list of names
        """
        return self._get_names(DiscreteParameter, self.input_space)

    def get_categorical_parameters(self) -> List[CategoricalParameter]:
        """Obtain all the categorical parameters

        Returns:
            List[CategoricalSpace]: space of categorical parameters
        """
        return self._get_parameters(CategoricalParameter, self.input_space)

    def get_categorical_names(self) -> List[str]:
        """Receive the names of all the categorical parameters

        Returns:
            List[str]: list of names
        """
        return self._get_names(CategoricalParameter, self.input_space)

    def _get_names(self, type: TypeVar, space: List[ParameterInterface]) -> List[str]:
        return [parameter.name for parameter in space if isinstance(parameter, type)]

    def _get_parameters(self, type: TypeVar, space: List[ParameterInterface]) -> List[ParameterInterface]:
        return list(
            filter(
                lambda parameter: isinstance(parameter, type),
                space,
            )
        )
