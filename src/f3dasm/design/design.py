#                                                                       Modules
# =============================================================================

# Standard
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Type, TypeVar

# Third-party core
import numpy as np
import pandas as pd
from hydra.utils import instantiate

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
    """Main class for defining design of experiments space.

    Parameters
    ----------
    input_space : Dict[str, Parameter], optional
        Dict of input parameters, by default an empty dict
    output_space : Dict[str, Parameter], optional
        Dict of output parameters, by default an empty dict
    constraints : List[Constraint], optional
        List of constraints, by default an empty list

    Raises
    ------
    ValueError
        If duplicate names are found in input or output names.
    """

    input_space: Dict[str, Parameter] = field(default_factory=dict)
    output_space: Dict[str, Parameter] = field(default_factory=dict)
    constraints: List[Constraint] = field(default_factory=list)

    @classmethod
    def from_json(cls: Type['DesignSpace'], json_string: str) -> 'DesignSpace':
        """
        Create a DesignSpace object from a JSON string.

        Parameters
        ----------
        json_string : str
            JSON string encoding the DesignSpace object.

        Returns
        -------
        DesignSpace
            The created DesignSpace object.
        """
        # Load JSON string
        design_dict = json.loads(json_string)
        return cls.from_dict(design_dict)

    @classmethod
    def from_yaml(cls: Type['DesignSpace'], yaml: Dict[str, Dict[str, Dict[str, Any]]]) -> 'DesignSpace':
        """Initializ a DesignSpace from a Hydra YAML configuration file


        Notes
        -----
        The YAML file should have the following structure:
        Two nested dictionaries where the dictionary denote the input_space 
        and output_space respectively.


        Parameters
        ----------
        yaml
            yaml dictionary

        Returns
        -------
            DesignSpace class
        """
        args = {}
        for space, params in yaml.items():
            args[space] = {name: instantiate(param) for name, param in params.items()}
        return cls(**args)

    @ classmethod
    def from_dict(cls: Type['DesignSpace'], design_dict: dict) -> 'DesignSpace':
        """
        Create a DesignSpace object from a dictionary.

        Parameters
        ----------
        design_dict : dict
            Dictionary representation of the information to construct the DesignSpace.

        Returns
        -------
        DesignSpace
            The created DesignSpace object.
        """
        for key, space in design_dict.items():
            parameters = {}
            for name, parameter in space.items():
                parameters[name] = Parameter.from_json(parameter)

            design_dict[key] = parameters

        return cls(**design_dict)

    def to_json(self) -> str:
        """Return JSON representation of the design space.

        Returns
        -------
        str
            JSON representation of the design space.
        """
        # Missing constraints
        args = {'input_space': {name: parameter.to_json() for name, parameter in self.input_space.items()},
                'output_space': {name: parameter.to_json() for name, parameter in self.output_space.items()},
                }
        return json.dumps(args)

    def create_empty_dataframe(self) -> pd.DataFrame:
        """Create an empty DataFrame with input and output space columns.

        Returns
        -------
        pd.DataFrame
            DataFrame containing "input" and "output" columns.
        """
        # input columns
        input_columns = [("input", name) for name, parameter in self.input_space.items()]

        df_input = pd.DataFrame(columns=input_columns).astype(
            self._cast_types_dataframe(self.input_space, label="input")
        )

        # Set the categories tot the categorical input parameters
        for name, categorical_input in self.get_categorical_input_parameters().items():
            df_input[('input', name)] = pd.Categorical(
                df_input[('input', name)], categories=categorical_input.categories)

        # output columns
        output_columns = [("output", name) for name, parameter in self.output_space.items()]

        df_output = pd.DataFrame(columns=output_columns).astype(
            self._cast_types_dataframe(self.output_space, label="output")
        )

        # Set the categories tot the categorical output parameters
        for name, categorical_output in self.get_categorical_output_parameters().items():
            df_output[('output', name)] = pd.Categorical(
                df_input[('output', name)], categories=categorical_output.categories)

        return pd.concat([df_input, df_output])

    def add_input_space(self, name: str, space: Parameter):
        """Add a new input parameter to the design space.

        Parameters
        ----------
        space : Parameter
            Input parameter to be added.
        """
        self.input_space[name] = space
        return

    def add_output_space(self, name: str, space: Parameter):
        """Add a new output parameter to the design space.

        Parameters
        ----------
        space : Parameter
            Output parameter to be added.
        """
        self.output_space[name] = space

    def get_input_space(self) -> Dict[str, Parameter]:
        """Return input parameters.

        Returns
        -------
        Dict[str, Parameter]
            List of input parameters.
        """
        return self.input_space

    def get_output_space(self) -> Dict[str, Parameter]:
        """Return output parameters.

        Returns
        -------
        Dict[str, Parameter]
            List of output parameters.
        """
        return self.output_space

    def get_output_names(self) -> List[str]:
        """Get the names of the output parameters

        Returns
        -------
            List of the names of the output parameters
        """
        # return [("output", name) for name, parameter in self.output_space.items()]
        return list(self.output_space.keys())

    def get_input_names(self) -> List[str]:
        """Get the names of the input parameters

        Returns
        -------
            List of the names of the input parameters
        """
        # return [("input", name) for name, parameter in self.input_space.items()]
        return list(self.input_space.keys())

    def is_single_objective_continuous(self) -> bool:
        """Checks whether the output of the model is a single continuous objective value.

        A model is considered to have a single continuous objective if all of
        its input and output parameters are continuous, and it returns only one output value.

        Returns
        -------
        bool
            True if the model's output is a single continuous objective value, False otherwise.
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

    def get_continuous_input_parameters(self) -> Dict[str, ContinuousParameter]:
        """Obtain all the continuous parameters

        Returns
        -------
            space of continuous parameters
        """
        return self.filter_parameters(ContinuousParameter).get_input_space()

    def get_continuous_input_names(self) -> List[str]:
        """Receive the continuous parameter names of the input space

        Returns
        -------
            list of names of the continuous input parameters
        """

        return self.filter_parameters(ContinuousParameter).get_input_names()

    def get_discrete_input_parameters(self) -> Dict[str, DiscreteParameter]:
        """Obtain all the discrete parameters

        Returns
        -------
            space of discrete parameters
        """
        return self.filter_parameters(DiscreteParameter).get_input_space()

    def get_discrete_input_names(self) -> List[str]:
        """Receive the names of all the discrete parameters

        Returns
        -------
            list of names
        """
        return self.filter_parameters(DiscreteParameter).get_input_names()

    def get_categorical_input_parameters(self) -> Dict[str, CategoricalParameter]:
        """Obtain all the categorical input parameters

        Returns
        -------
            space of categorical input parameters
        """
        return self.filter_parameters(CategoricalParameter).get_input_space()

    def get_categorical_output_parameters(self) -> Dict[str, CategoricalParameter]:
        """Obtain all the categorical output parameters

        Returns
        -------
            space of categorical output parameters
        """
        return self.filter_parameters(CategoricalParameter).get_output_space()

    def get_categorical_input_names(self) -> List[str]:
        """Receive the names of the categorical input parameters

        Returns
        -------
            list of names of categorical input parameters
        """
        return self.filter_parameters(CategoricalParameter).get_input_names()

    def get_constant_input_parameters(self) -> Dict[str, ConstantParameter]:
        """Obtain all the constant input parameters

        Returns
        -------
            space of constant input parameters
        """
        return self.filter_parameters(ConstantParameter).get_input_space()

    def get_constant_input_names(self) -> List[str]:
        """Receive the names of the constant input parameters

        Returns
        -------
            list of names of constant input parameters
        """
        return self.filter_parameters(ConstantParameter).get_input_names()

    def get_bounds(self) -> np.ndarray:
        """Return the boundary constraints of the continuous input parameters

        Returns
        -------
            numpy array with lower and upper bound for each continuous inpu dimension
        """
        return np.array(
            [[parameter.lower_bound, parameter.upper_bound]
                for _, parameter in self.get_continuous_input_parameters().items()]
        )

    def filter_parameters(self, type: Type[Parameter]) -> 'DesignSpace':
        """Filter the parameters of the design space by type

        Parameters
        ----------
        type : Type[Parameter]
            Type of the parameters to be filtered

        Returns
        -------
        DesignSpace
            Design space with the filtered parameters
        """
        return DesignSpace(
            input_space={name: parameter for name, parameter in self.input_space.items()
                         if isinstance(parameter, type)},
            output_space={name: parameter for name, parameter in self.output_space.items()
                          if isinstance(parameter, type)},
        )

    def _cast_types_dataframe(self, space: Dict[str, Parameter], label: str) -> dict:
        """Make a dictionary that provides the datatype of each parameter"""
        return {(label, name): parameter._type for name, parameter in space.items()}

    def _all_input_continuous(self) -> bool:
        """Check if all input parameters are continuous"""
        return self.get_number_of_input_parameters() \
            == self.filter_parameters(ContinuousParameter).get_number_of_input_parameters()

    def _all_output_continuous(self) -> bool:
        """Check if all output parameters are continuous"""

        return self.get_number_of_output_parameters() \
            == self.filter_parameters(ContinuousParameter).get_number_of_output_parameters()


def make_nd_continuous_design(bounds: np.ndarray, dimensionality: int) -> DesignSpace:
    """Create a continuous design space with a single-objective continuous output.

    Parameters
    ----------
    bounds : numpy.ndarray
        A 2D numpy array of shape (dimensionality, 2) specifying the lower and upper bounds of every dimension.
    dimensionality : int
        The number of dimensions.

    Returns
    -------
    DesignSpace
        A continuous design space with a single-objective continuous output.

    Notes
    -----
    This function creates a DesignSpace object consisting of continuous input parameters and a single continuous
    output parameter. The lower and upper bounds of each input dimension are specified in the `bounds` parameter.
    The input parameters are named "x0", "x1" .. "The output parameter is named "y".

    Example
    -------
    >>> bounds = np.array([[-5.0, 5.0], [-2.0, 2.0]])
    >>> dimensionality = 2
    >>> design_space = make_nd_continuous_design(bounds, dimensionality)
    """
    input_space, output_space = {}, {}
    for dim in range(dimensionality):
        input_space[f"x{dim}"] = ContinuousParameter(lower_bound=bounds[dim, 0], upper_bound=bounds[dim, 1])

    output_space["y"] = ContinuousParameter()

    return DesignSpace(input_space=input_space, output_space=output_space)
