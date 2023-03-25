#                                                                       Modules
# =============================================================================

# Standard
import json
from dataclasses import dataclass, field
from typing import List, TypeVar

# Third-party core
import numpy as np
import pandas as pd

# Local
from .constraint import Constraint
from .parameter import (CategoricalParameter, ConstantParameter,
                        ContinuousParameter, DiscreteParameter, Parameter)

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


@dataclass
class DesignSpace:
    """Main class for the design of experiments space


    Parameters
    ----------
    input_space
        list of input parameters
    output_space
        list of output parameters
    constraints
        list of constraints
    """

    input_space: List[Parameter] = field(default_factory=list)
    output_space: List[Parameter] = field(default_factory=list)
    constraints: List[Constraint] = field(default_factory=list)

    def __post_init__(self):
        self._check_names()

    def _check_names(self):
        if len(self.get_input_names()) != len(set(self.get_input_names())):
            raise ValueError("Duplicate names found in input names!")

        if len(self.get_output_names()) != len(set(self.get_output_names())):
            raise ValueError("Duplicate names found in output names!")

    def to_json(self) -> str:
        # Missing constraints
        args = {'input_space': [parameter.to_json() for parameter in self.input_space],
                'output_space': [parameter.to_json() for parameter in self.output_space]
                }
        return json.dumps(args)

    def get_empty_dataframe(self) -> pd.DataFrame:
        """Create an empty DataFrame with the information of the input and output space

        Returns
        -------
            DataFrame containing "input" and "output" columns
        """
        # input columns
        df_input = pd.DataFrame(columns=self.get_input_names()).astype(
            self._cast_types_dataframe(self.input_space, label="input")
        )

        # output columns
        df_output = pd.DataFrame(columns=self.get_output_names()).astype(
            self._cast_types_dataframe(self.output_space, label="output")
        )

        return pd.concat([df_input, df_output])

    def add_input_space(self, space: Parameter):
        """Add a new parameter to the searchspace

        Parameters
        ----------
        space
            search space parameter to be added
        """
        self.input_space.append(space)
        return

    def add_output_space(self, space: Parameter):
        """Add a new parameter to the searchspace

        Parameters
        ----------
        space
            search space parameter to be added
        """
        self.output_space.append(space)

    def get_input_space(self) -> List[Parameter]:
        """Get the input space

        Returns
        -------
            List of input parameters
        """
        return self.input_space

    def get_output_space(self) -> List[Parameter]:
        """Get the output space

        Returns
        -------
            List of output parameters
        """
        return self.output_space

    def get_output_names(self) -> List[str]:
        """Get the names of the output parameters

        Returns
        -------
            List of the names of the output parameters
        """
        return [("output", s.name) for s in self.output_space]

    def get_input_names(self) -> List[str]:
        """Get the names of the input parameters

        Returns
        -------
            List of the names of the input parameters
        """
        return [("input", s.name) for s in self.input_space]

    def is_single_objective_continuous(self) -> bool:
        """Check if the output is single objective and continuous

        Returns
        -------
            boolean value if the above described condition is true
        """
        return (
            self._all_input_continuous()
            and self._all_output_continuous()
            and self.get_number_of_output_parameters() == 1
        )

    def get_number_of_input_parameters(self) -> int:
        """Obtain the number of input parameters

        Returns
        -------
            number of input parameters
        """
        return len(self.input_space)

    def get_number_of_output_parameters(self) -> int:
        """Obtain the number of output parameters

        Returns
        -------
            number of output parameters
        """
        return len(self.output_space)

    def get_continuous_input_parameters(self) -> List[ContinuousParameter]:
        """Obtain all the continuous parameters

        Returns
        -------
            space of continuous parameters
        """
        return self._get_parameters(ContinuousParameter, self.input_space)

    def get_continuous_input_names(self) -> List[str]:
        """Receive the continuous parameter names of the input space

        Returns
        -------
            list of names of the continuous input parameters
        """
        return self._get_names(ContinuousParameter, self.input_space)

    def get_discrete_input_parameters(self) -> List[DiscreteParameter]:
        """Obtain all the discrete parameters

        Returns
        -------
            space of discrete parameters
        """
        return self._get_parameters(DiscreteParameter, self.input_space)

    def get_discrete_input_names(self) -> List[str]:
        """Receive the names of all the discrete parameters

        Returns
        -------
            list of names
        """
        return self._get_names(DiscreteParameter, self.input_space)

    def get_categorical_input_parameters(self) -> List[CategoricalParameter]:
        """Obtain all the categorical input parameters

        Returns
        -------
            space of categorical input parameters
        """
        return self._get_parameters(CategoricalParameter, self.input_space)

    def get_categorical_input_names(self) -> List[str]:
        """Receive the names of the categorical input parameters

        Returns
        -------
            list of names of categorical input parameters
        """
        return self._get_names(CategoricalParameter, self.input_space)

    def get_constant_input_parameters(self) -> List[ConstantParameter]:
        """Obtain all the constant input parameters

        Returns
        -------
            space of constant input parameters
        """
        return self._get_parameters(ConstantParameter, self.input_space)

    def get_constant_input_names(self) -> List[str]:
        """Receive the names of the constant input parameters

        Returns
        -------
            list of names of constant input parameters
        """
        return self._get_names(ConstantParameter, self.input_space)

    def get_bounds(self) -> np.ndarray:
        """Return the boundary constraints of the continuous input parameters

        Returns
        -------
            numpy array with lower and upper bound for each continuous inpu dimension
        """
        return np.array(
            [[parameter.lower_bound, parameter.upper_bound]
                for parameter in self.get_continuous_input_parameters()]
        )

    def _get_names(self, type: TypeVar, space: List[Parameter]) -> List[str]:
        return [parameter.name for parameter in space if isinstance(parameter, type)]

    def _get_parameters(self, type: TypeVar, space: List[Parameter]) -> List[Parameter]:
        return list(
            filter(
                lambda parameter: isinstance(parameter, type),
                space,
            )
        )

    def _cast_types_dataframe(self, space: List[Parameter], label: str) -> dict:
        """Make a dictionary that provides the datatype of each parameter"""
        return {(label, parameter.name): parameter._type for parameter in space}

    def _check_space_on_type(self, type: TypeVar, space: List[Parameter]) -> bool:
        return all(isinstance(parameter, type) for parameter in space)

    def _all_input_continuous(self) -> bool:
        """Check if all input parameters are continuous"""
        return self._check_space_on_type(ContinuousParameter, self.input_space)

    def _all_output_continuous(self) -> bool:
        """Check if all output parameters are continuous"""
        return self._check_space_on_type(ContinuousParameter, self.output_space)


def make_nd_continuous_design(bounds: np.ndarray, dimensionality: int) -> DesignSpace:
    """Helper function to make an continuous design space with a single-objective continuous output

    Parameters
    ----------
    bounds
        lower and upper bounds of every dimension
    dimensionality
        number of dimensions

    Returns
    -------
        continuous, single-objective designspace
    """
    input_space, output_space = [], []
    for dim in range(dimensionality):
        input_space.append(ContinuousParameter(
            name=f"x{dim}", lower_bound=bounds[dim, 0], upper_bound=bounds[dim, 1]))

    output_space.append(ContinuousParameter(name="y"))

    return DesignSpace(input_space=input_space, output_space=output_space)
