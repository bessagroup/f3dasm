"""
The Domain is a set of Parameter instances that make up
 the feasible search space.
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
import math
import pickle
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Type

if sys.version_info < (3, 8):  # NOQA
    from typing_extensions import Literal  # NOQA
else:
    from typing import Literal

# Third-party core
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

# Local
from .parameter import (CategoricalType, _CategoricalParameter,
                        _ConstantParameter, _ContinuousParameter,
                        _DiscreteParameter, _OutputParameter, _Parameter)

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


@dataclass
class Domain:
    """Main class for defining the domain of the design of experiments.

    Parameters
    ----------
    space : Dict[str, Parameter], optional
        Dict of input parameters, by default an empty dict
    """

    space: Dict[str, _Parameter] = field(default_factory=dict)
    output_space: Dict[str, _OutputParameter] = field(default_factory=dict)

    def __len__(self) -> int:
        """The len() method returns the number of parameters"""
        return len(self.space)

    def __eq__(self, other: Domain) -> bool:
        """Custom equality comparison for Domain objects."""

        if not isinstance(other, Domain):
            return TypeError(f"Cannot compare Domain with \
                {type(other.__name__)}")
        return (
            self.space == other.space)

    def items(self) -> Iterator[_Parameter]:
        """Return an iterator over the items of the parameters"""
        return self.space.items()

    def values(self) -> Iterator[_Parameter]:
        """Return an iterator over the values of the parameters"""
        return self.space.values()

    def keys(self) -> Iterator[str]:
        """Return an iterator over the keys of the parameters"""
        return self.space.keys()

    @property
    def names(self) -> List[str]:
        """Return a list of the names of the parameters"""
        return list(self.keys())

    @property
    def output_names(self) -> List[str]:
        """Return a list of the names of the output parameters"""
        return list(self.output_space.keys())

    @property
    def continuous(self) -> Domain:
        """Returns a Domain object containing only the continuous parameters"""
        return self._filter(_ContinuousParameter)

    @property
    def discrete(self) -> Domain:
        """Returns a Domain object containing only the discrete parameters"""
        return self._filter(_DiscreteParameter)

    @property
    def categorical(self) -> Domain:
        """Returns a Domain object containing only
         the categorical parameters"""
        return self._filter(_CategoricalParameter)

    @property
    def constant(self) -> Domain:
        """Returns a Domain object containing only the constant parameters"""
        return self._filter(_ConstantParameter)
#                                                      Alternative constructors
# =============================================================================

    @classmethod
    def from_file(cls: Type[Domain], filename: Path | str) -> Domain:
        """Create a Domain object from a pickle file.

        Parameters
        ----------
        filename : Path
            Name of the file.

        Returns
        -------
        Domain
            Domain object containing the loaded data.
        """
        # convert filename to Path object
        filename = Path(filename)

        # Check if filename exists
        if not filename.with_suffix('.pkl').exists():
            raise FileNotFoundError(f"Domain file {filename} does not exist.")

        with open(filename.with_suffix('.pkl'), "rb") as file:
            obj = pickle.load(file)

        return obj

    @classmethod
    def from_yaml(cls: Type[Domain], cfg: DictConfig) -> Domain:
        """Initialize a Domain from a Hydra YAML configuration file key


        Note
        ----
        The YAML file should have the following structure:

        .. code-block:: yaml

            domain:
                <parameter_name>:
                    type: <parameter_type>
                    <parameter_type_specific_parameters>
                <parameter_name>:
                    type: <parameter_type>
                    <parameter_type_specific_parameters>


        Parameters
        ----------
        cfg : DictConfig
            YAML dictionary key of the domain.

        Returns
        -------
        Domain
            Domain object
        """
        domain = cls()

        for key, value in cfg.items():
            _dict = OmegaConf.to_container(value, resolve=True)
            domain.add(name=key, type=_dict.pop('type'), **_dict)

        return domain

    @classmethod
    def from_dataframe(cls, df_input: pd.DataFrame,
                       df_output: pd.DataFrame) -> Domain:
        """Initializes a Domain from a pandas DataFrame.

        Parameters
        ----------
        df_input : pd.DataFrame
            DataFrame containing the input parameters.
        df_output : pd.DataFrame
            DataFrame containing the output parameters.

        Returns
        -------
        Domain
            Domain object
        """
        input_space = {}
        for name, type in df_input.dtypes.items():
            if type == 'float64':
                if float(df_input[name].min()) == float(df_input[name].max()):
                    input_space[name] = _ConstantParameter(
                        value=float(df_input[name].min()))
                    continue

                input_space[name] = _ContinuousParameter(lower_bound=float(
                    df_input[name].min()),
                    upper_bound=float(df_input[name].max()))
            elif type == 'int64':
                if int(df_input[name].min()) == int(df_input[name].max()):
                    input_space[name] = _ConstantParameter(
                        value=int(df_input[name].min()))
                    continue

                input_space[name] = _DiscreteParameter(lower_bound=int(
                    df_input[name].min()),
                    upper_bound=int(df_input[name].max()))
            else:
                input_space[name] = _CategoricalParameter(
                    df_input[name].unique().tolist())

        output_space = {}
        for name in df_output.columns:
            output_space[name] = _OutputParameter(to_disk=False)

        return cls(space=input_space, output_space=output_space)

#                                                                        Export
# =============================================================================

    def store(self, filename: Path) -> None:
        """Stores the Domain in a pickle file.

        Parameters
        ----------
        filename : str
            Name of the file.
        """
        with open(filename.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(self, f)

    def _cast_types_dataframe(self) -> dict:
        """Make a dictionary that provides the datatype of each parameter"""
        return {name: parameter._type for
                name, parameter in self.space.items()}

    def _create_empty_dataframe(self) -> pd.DataFrame:
        """Create an empty DataFrame with input columns.

        Returns
        -------
        pd.DataFrame
            DataFrame containing "input" columns.
        """
        # input columns
        input_columns = [name for name in self.space.keys()]

        return pd.DataFrame(columns=input_columns).astype(
            self._cast_types_dataframe()
        )

#                                                  Append and remove parameters
# =============================================================================

    def _add(self, name: str, parameter: _Parameter):
        # Check if parameter is already in the domain
        if name in self.space:
            raise KeyError(
                f"Parameter {name} already exists in the domain! \
                     Choose a different name.")

        self.space[name] = parameter

    def add_int(self, name: str, low: int, high: int, step: int = 1):
        """Add a new discrete input parameter to the domain.

        Parameters
        ----------
        name : str
            Name of the input parameter.
        low : int
            Lower bound of the input parameter.
        high : int
            Upper bound of the input parameter.
        step : int, optional
            Step size of the input parameter, by default 1.

        Example
        -------
        >>> domain = Domain()
        >>> domain.add_int('param1', 0, 10, 2)
        >>> domain.space
        {'param1': _DiscreteParameter(lower_bound=0, upper_bound=10, step=2)}

        Note
        ----
        If the lower and upper bound are equal, then a constant parameter
        will be added to the domain!
        """
        if low == high:
            self.add_constant(name, low)
        self._add(name, _DiscreteParameter(low, high, step))

    def add_float(self, name: str, low: float = -np.inf, high: float = np.inf,
                  log: bool = False):
        """Add a new continuous input parameter to the domain.

        Parameters
        ----------
        name : str
            Name of the input parameter.
        low : float, optional
            Lower bound of the input parameter. By default -np.inf.
        high : float
            Upper bound of the input parameter. By default np.inf.
        log : bool, optional
            Whether to use a logarithmic scale, by default False.

        Example
        -------
        >>> domain = Domain()
        >>> domain.add_float('param1', 0., 10., log=True)
        >>> domain.space
        {'param1': _ContinuousParameter(lower_bound=0.,
         upper_bound=10., log=True)}

        Note
        ----
        If the lower and upper bound are equal, then a constant parameter
        will be added to the domain!
        """
        if math.isclose(low, high):
            self.add_constant(name, low)
        else:
            self._add(name, _ContinuousParameter(low, high, log))

    def add_category(self, name: str, categories: Sequence[CategoricalType]):
        """Add a new categorical input parameter to the domain.

        Parameters
        ----------
        name : str
            Name of the input parameter.
        categories : List[Any]
            Categories of the input parameter.

        Example
        -------
        >>> domain = Domain()
        >>> domain.add_category('param1', [0, 1, 2])
        >>> domain.space
        {'param1': CategoricalParameter(categories=[0, 1, 2])}
        """
        self._add(name, _CategoricalParameter(categories))

    def add_constant(self, name: str, value: Any):
        """Add a new constant input parameter to the domain.

        Parameters
        ----------
        name : str
            Name of the input parameter.
        value : Any
            Value of the input parameter.

        Example
        -------
        >>> domain = Domain()
        >>> domain.add_constant('param1', 0)
        >>> domain.space
        {'param1': ConstantParameter(value=0)}
        """
        self._add(name, _ConstantParameter(value))

    def add_parameter(self, name: str):
        """Add a new parameter to the domain.

        Parameters
        ----------
        name : str
            Name of the input parameter.

        Example
        -------
        >>> domain = Domain()
        >>> domain.add_parameter('param1')
        >>> domain.space
        {'param1': Parameter()}
        """
        self._add(name, _Parameter())

    def add(self, name: str,
            type: Literal['float', 'int', 'category', 'constant'],
            **kwargs):
        """Add a new input parameter to the domain.

        Parameters
        ----------

        name : str
            Name of the input parameter.
        type : Literal['float', 'int', 'category', 'constant']
            Type of the input parameter.
        **kwargs
            Keyword arguments for the input parameter.

        Raises
        ------
        ValueError
            If the type is not known.

        Example
        -------
        >>> domain = Domain()
        >>> domain._new_add('param1', 'float', low=0., high=1.)
        >>> domain.space
        {'param1': _ContinuousParameter(lower_bound=0., upper_bound=1.)}
        """

        if type == 'float':
            self.add_float(name, **kwargs)
        elif type == 'int':
            self.add_int(name, **kwargs)
        elif type == 'category':
            self.add_category(name, **kwargs)
        elif type == 'constant':
            self.add_constant(name, **kwargs)
        else:
            raise ValueError(
                f"Unknown type {type}!"
                f"Possible types are: 'float', 'int', 'category', 'constant'.")

    def add_output(self, name: str, to_disk: bool):
        """Add a new output parameter to the domain.

        Parameters
        ----------
        name : str
            Name of the output parameter.
        to_disk : bool
            Whether to store the output parameter on disk.

        Example
        -------
        >>> domain = Domain()
        >>> domain.add_output('param1', True)
        >>> domain.space
        {'param1': OutputParameter(to_disk=True)}
        """
        if name in self.output_space:
            raise KeyError(
                f"Parameter {name} already exists in the domain! \
                     Choose a different name.")

        self.output_space[name] = _OutputParameter(to_disk)
#                                                                       Getters
# =============================================================================

    def get_continuous_parameters(self) -> Dict[str, _ContinuousParameter]:
        """Get all continuous input parameters.

        Returns
        -------
        Dict[str, _ContinuousParameter]
            Space of continuous input parameters.

        Example
        -------
        >>> domain = Domain()
        >>> domain.space = {
        ...     'param1': _ContinuousParameter(lower_bound=0., upper_bound=1.),
        ...     'param2': CategoricalParameter(categories=['A', 'B', 'C']),
        ...     'param3': _ContinuousParameter(lower_bound=2., upper_bound=5.)
        ... }
        >>> continuous_input_params = domain.get_continuous_input_parameters()
        >>> continuous_input_params
        {'param1': _ContinuousParameter(lower_bound=0., upper_bound=1.),
         'param3': _ContinuousParameter(lower_bound=2., upper_bound=5.)}
        """
        return self._filter(_ContinuousParameter).space

    def get_continuous_names(self) -> List[str]:
        """Get the names of continuous input parameters in the input space.

        Returns
        -------
        List[str]
            List of names of continuous input parameters.

        Example
        -------
        >>> domain = Domain()
        >>> domain.space = {
        ...     'param1': _ContinuousParameter(lower_bound=0., upper_bound=1.),
        ...     'param2': _DiscreteParameter(lower_bound=1, upper_bound=3),
        ...     'param3': _ContinuousParameter(lower_bound=2., upper_bound=5.)
        ... }
        >>> continuous_input_names = domain.get_continuous_input_names()
        >>> continuous_input_names
        ['param1', 'param3']
        """
        return self._filter(_ContinuousParameter).names

    def get_discrete_parameters(self) -> Dict[str, _DiscreteParameter]:
        """Retrieve all discrete input parameters.

        Returns
        -------
        Dict[str, _DiscreteParameter]
            Space of discrete input parameters.

        Example
        -------
        >>> domain = Domain()
        >>> domain.space = {
        ...     'param1': _DiscreteParameter(lower_bound=1, upperBound=4),
        ...     'param2': CategoricalParameter(categories=['A', 'B', 'C']),
        ...     'param3': _DiscreteParameter(lower_bound=4, upperBound=6)
        ... }
        >>> discrete_input_params = domain.get_discrete_input_parameters()
        >>> discrete_input_params
        {'param1': _DiscreteParameter(lower_bound=1, upperBound=4)),
         'param3': _DiscreteParameter(lower_bound=4, upperBound=6)}
        """
        return self._filter(_DiscreteParameter).space

    def get_discrete_names(self) -> List[str]:
        """Retrieve the names of all discrete input parameters.

        Returns
        -------
        List[str]
            List of names of discrete input parameters.

        Example
        -------
        >>> domain = Domain()
        >>> domain.space = {
        ...     'param1': _DiscreteParameter(lower_bound=1, upperBound=4),
        ...     'param2': _ContinuousParameter(lower_bound=0, upper_bound=1),
        ...     'param3': _DiscreteParameter(lower_bound=4, upperBound=6)
        ... }
        >>> discrete_input_names = domain.get_discrete_input_names()
        >>> discrete_input_names
        ['param1', 'param3']
        """
        return self._filter(_DiscreteParameter).names

    def get_categorical_parameters(self) -> Dict[str, _CategoricalParameter]:
        """Retrieve all categorical input parameters.

        Returns
        -------
        Dict[str, CategoricalParameter]
            Space of categorical input parameters.

        Example
        -------
        >>> domain = Domain()
        >>> domain.space = {
        ...     'param1': CategoricalParameter(categories=['A', 'B', 'C']),
        ...     'param2': _ContinuousParameter(lower_bound=0, upper_bound=1),
        ...     'param3': CategoricalParameter(categories=['X', 'Y', 'Z'])
        ... }
        >>> categorical_input_params =
         domain.get_categorical_input_parameters()
        >>> categorical_input_params
        {'param1': CategoricalParameter(categories=['A', 'B', 'C']),
         'param3': CategoricalParameter(categories=['X', 'Y', 'Z'])}
        """
        return self._filter(_CategoricalParameter).space

    def get_categorical_names(self) -> List[str]:
        """Retrieve the names of categorical input parameters.

        Returns
        -------
        List[str]
            List of names of categorical input parameters.

        Example
        -------
        >>> domain = Domain()
        >>> domain.space = {
        ...     'param1': CategoricalParameter(categories=['A', 'B', 'C']),
        ...     'param2': _ContinuousParameter(lower_bound=0, upper_bound=1),
        ...     'param3': CategoricalParameter(categories=['X', 'Y', 'Z'])
        ... }
        >>> categorical_input_names = domain.get_categorical_input_names()
        >>> categorical_input_names
        ['param1', 'param3']
        """
        return self._filter(_CategoricalParameter).names

    def get_constant_parameters(self) -> Dict[str, _ConstantParameter]:
        """Retrieve all constant input parameters.

        Returns
        -------
        Dict[str, ConstantParameter]
            Space of constant input parameters.

        Example
        -------
        >>> domain = Domain()
        >>> domain.space = {
        ...     'param1': ConstantParameter(value=0),
        ...     'param2': CategoricalParameter(categories=['A', 'B', 'C']),
        ...     'param3': ConstantParameter(value=1)
        ... }
        >>> constant_input_params = domain.get_constant_input_parameters()
        >>> constant_input_params
        {'param1': ConstantParameter(value=0),
         'param3': ConstantParameter(value=1)}
        """
        return self._filter(_ConstantParameter).space

    def get_constant_names(self) -> List[str]:
        """Receive the names of the constant input parameters

        Returns
        -------
            list of names of constant input parameters

        Example
        -------
        >>> domain = Domain()
        >>> domain.space = {
        ...     'param1': ConstantParameter(value=0),
        ...     'param2': ConstantParameter(value=1),
        ...     'param3': _ContinuousParameter(lower_bound=0, upper_bound=1)
        ... }
        >>> constant_input_names = domain.get_constant_input_names()
        >>> constant_input_names
        ['param1', 'param2']
        """
        return self._filter(_ConstantParameter).names

    def get_bounds(self) -> np.ndarray:
        """Return the boundary constraints of the continuous input parameters

        Returns
        -------
            numpy array with lower and upper bound for each \
            continuous input dimension

        Example
        -------
        >>> domain = Domain()
        >>> domain.space = {
        ...     'param1': _ContinuousParameter(lower_bound=0, upper_bound=1),
        ...     'param2': _ContinuousParameter(lower_bound=-1, upper_bound=1),
        ...     'param3': _ContinuousParameter(lower_bound=0, upper_bound=10)
        ... }
        >>> bounds = domain.get_bounds()
        >>> bounds
        array([[ 0.,  1.],
            [-1.,  1.],
            [ 0., 10.]])
        """
        return np.array(
            [[parameter.lower_bound, parameter.upper_bound]
                for _, parameter in self.get_continuous_parameters().items()]
        )

    def _filter(self, type: Type[_Parameter]) -> Domain:
        """Filter the parameters of the domain by type

        Parameters
        ----------
        type : Type[Parameter]
            Type of the parameters to be filtered

        Returns
        -------
        Domain
            Domain with the filtered parameters

        Example
        -------
        >>> domain = Domain()
        >>> domain.space = {
        ...     'param1': _ContinuousParameter(lower_bound=0., upper_bound=1.),
        ...     'param2': _DiscreteParameter(lower_bound=0, upper_bound=8),
        ...     'param3': CategoricalParameter(categories=['cat1', 'cat2'])
        ... }
        >>> filtered_domain = domain.filter_parameters(_ContinuousParameter)
        >>> filtered_domain.space
        {'param1': _ContinuousParameter(lower_bound=0, upper_bound=1)}

        """
        return Domain(
            space={name: parameter for name, parameter in self.space.items()
                   if isinstance(parameter, type)}
        )

    def select(self, names: str | Iterable[str]) -> Domain:
        """Select a subset of parameters from the domain.

        Parameters
        ----------

        names : str or Iterable[str]
            The names of the parameters to select.

        Returns
        -------
        Domain
            A new domain with the selected parameters.

        Example
        -------
        >>> domain = Domain()
        >>> domain.space = {
        ...     'param1': _ContinuousParameter(lower_bound=0., upper_bound=1.),
        ...     'param2': _DiscreteParameter(lower_bound=0, upper_bound=8),
        ...     'param3': CategoricalParameter(categories=['cat1', 'cat2'])
        ... }
        >>> domain.select(['param1', 'param3'])
        Domain({'param1': _ContinuousParameter(lower_bound=0, upper_bound=1),
                'param3': CategoricalParameter(categories=['cat1', 'cat2'])})
        """

        if isinstance(names, str):
            names = [names]

        return Domain(space={key: self.space[key] for key in names})

#                                                                 Miscellaneous
# =============================================================================

    def _all_input_continuous(self) -> bool:
        """Check if all input parameters are continuous"""
        return len(self) == len(self._filter(_ContinuousParameter))

    def _check_output(self, names: List[str]):
        """Check if output is in the domain and add it if not

        Parameters
        ----------

        names : list of str
            Names of the outputs to be checked

        Example
        -------
        >>> domain = Domain()
        >>> domain.add_output('output1')
        >>> domain.add_output('output2')
        >>> domain._check_output(['output1', 'output2', 'output3'])
        >>> domain.output_space
        {'output1': _ContinuousParameter(lower_bound=-inf, upper_bound=inf),
         'output2': _ContinuousParameter(lower_bound=-inf, upper_bound=inf),
         'output3': _ContinuousParameter(lower_bound=-inf, upper_bound=inf)}
        """
        for output_name in names:
            if not self.is_in_output(output_name):
                print(f"Output {output_name} not in domain. Adding it.")
                self.add_output(output_name, to_disk=False)

    def is_in_output(self, output_name: str) -> bool:
        """Check if output is in the domain

        Parameters
        ----------
        output_name : str
            Name of the output

        Returns
        -------
        bool
            True if output is in the domain, False otherwise

        Example
        -------
        >>> domain = Domain()
        >>> domain.add_output('output1')
        >>> domain.is_in_output('output1')
        True
        >>> domain.is_in_output('output2')
        False
        """
        return output_name in self.output_space


def make_nd_continuous_domain(bounds: np.ndarray | List[List[float]],
                              dimensionality: int) -> Domain:
    """Create a continuous domain.

    Parameters
    ----------
    bounds : numpy.ndarray
        A 2D numpy array of shape (dimensionality, 2) specifying the lower \
        and upper bounds of every dimension.
    dimensionality : int
        The number of dimensions.

    Returns
    -------
    Domain
        A continuous domain with a continuous input.

    Note
    ----
    This function creates a Domain object consisting of \
    continuous input parameters.

    The lower and upper bounds of each input dimension are specified \
    in the `bounds` parameter.

    The input parameters are named "x0", "x1" ..

    Example
    -------
    >>> bounds = np.array([[-5.0, 5.0], [-2.0, 2.0]])
    >>> dimensionality = 2
    >>> domain = make_nd_continuous_domain(bounds, dimensionality)
    """
    space = {}

    # bounds is a list of lists, convert to numpy array:
    bounds = np.array(bounds)

    for dim in range(dimensionality):
        space[f"x{dim}"] = _ContinuousParameter(
            lower_bound=bounds[dim, 0], upper_bound=bounds[dim, 1])

    return Domain(space)


def _domain_factory(domain: Domain | DictConfig | None,
                    input_data: pd.DataFrame,
                    output_data: pd.DataFrame) -> Domain:
    if isinstance(domain, Domain):
        # domain._check_output(output_data.columns)
        return domain

    elif isinstance(domain, (Path, str)):
        return Domain.from_file(Path(domain))

    elif isinstance(domain, DictConfig):
        return Domain.from_yaml(domain)

    elif (input_data.empty and output_data.empty and domain is None):
        return Domain()

    elif domain is None:
        return Domain.from_dataframe(
            input_data, output_data)

    else:
        raise TypeError(
            f"Domain must be of type Domain, DictConfig "
            f"or None, not {type(domain)}")
