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
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Type

# Third-party core
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

# Local
from .parameter import (CategoricalParameter, CategoricalType,
                        ConstantParameter, ContinuousParameter,
                        DiscreteParameter, Parameter)

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


class Domain:
    """Main class for defining the domain of the design of experiments.

    Parameters
    ----------
    input_space : Dict[str, Parameter], optional
        Dict of input parameters, by default None
    output_space : Dict[str, Parameter], optional
        Dict of output parameters, by default None
    """

    def __init__(self, input_space: Dict[str, Parameter] = None,
                 output_space: Dict[str, Parameter] = None):
        self.input_space = input_space if input_space is not None else {}
        self.output_space = output_space if output_space is not None else {}

    def __len__(self) -> int:
        """The len() method returns the number of parameters"""
        return len(self.input_space)

    def __eq__(self, __o: Domain) -> bool:
        """Custom equality comparison for Domain objects."""

        if not isinstance(__o, Domain):
            return TypeError(f"Cannot compare Domain with \
                {type(__o.__name__)}")
        return (
            self.input_space == __o.input_space)

    def __str__(self):
        input_space_str = ", ".join(
            f"{k}: {v}" for k, v in self.input_space.items())
        output_space_str = ", ".join(
            f"{k}: {v}" for k, v in self.output_space.items())
        return (f"Domain(\n"
                f"  Input Space: {{ {input_space_str} }}\n"
                f"  Output Space: {{ {output_space_str} }}\n)")

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"input_space={repr(self.input_space)}, "
                f"output_space={repr(self.output_space)})")

    def __add__(self, __o: Domain) -> Domain:
        if not isinstance(__o, Domain):
            raise TypeError(f"Cannot add Domain with {type(__o.__name__)}")

        combined_space = {}
        # Merge values for keys that are present in both dictionaries
        for key in self.input_space.keys():
            if key in __o.input_space:
                combined_space[key] = self.input_space[key] + \
                    __o.input_space[key]
            else:
                combined_space[key] = self.input_space[key]

        # Add keys from dict2 that are not present in dict1
        for key in __o.input_space.keys():
            if key not in self.input_space:
                combined_space[key] = __o.input_space[key]

        return Domain(input_space=combined_space,
                      output_space={**self.output_space, **__o.output_space})

    @property
    def input_names(self) -> List[str]:
        """Return a list of the names of the parameters"""
        return list(self.input_space.keys())

    @property
    def output_names(self) -> List[str]:
        """Return a list of the names of the output parameters"""
        return list(self.output_space.keys())

    @property
    def continuous(self) -> Domain:
        """Returns a Domain object containing only the continuous parameters"""
        return self._filter(ContinuousParameter)

    @property
    def discrete(self) -> Domain:
        """Returns a Domain object containing only the discrete parameters"""
        return self._filter(DiscreteParameter)

    @property
    def categorical(self) -> Domain:
        """Returns a Domain object containing only
         the categorical parameters"""
        return self._filter(CategoricalParameter)

    @property
    def constant(self) -> Domain:
        """Returns a Domain object containing only the constant parameters"""
        return self._filter(ConstantParameter)
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
                    input_space[name] = ConstantParameter(
                        value=float(df_input[name].min()))
                    continue

                input_space[name] = ContinuousParameter(lower_bound=float(
                    df_input[name].min()),
                    upper_bound=float(df_input[name].max()))
            elif type == 'int64':
                if int(df_input[name].min()) == int(df_input[name].max()):
                    input_space[name] = ConstantParameter(
                        value=int(df_input[name].min()))
                    continue

                input_space[name] = DiscreteParameter(lower_bound=int(
                    df_input[name].min()),
                    upper_bound=int(df_input[name].max()))
            else:
                input_space[name] = CategoricalParameter(
                    df_input[name].unique().tolist())

        output_space = {}
        for name in df_output.columns:
            output_space[name] = Parameter(to_disk=False)

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

    # Not used
    def _cast_types_dataframe(self) -> dict:
        """Make a dictionary that provides the datatype of each parameter"""
        return {name: parameter._type for
                name, parameter in self.input_space.items()}

#                                                  Append and remove parameters
# =============================================================================

    def _add(self, name: str, parameter: Parameter):
        # Check if parameter is already in the domain
        if name in self.input_space:
            raise KeyError(
                f"Parameter {name} already exists in the domain! \
                     Choose a different name.")

        self.input_space[name] = parameter

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
        self._add(name, DiscreteParameter(low, high, step))

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
            self._add(name, ContinuousParameter(low, high, log))

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
        self._add(name, CategoricalParameter(categories))

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
        self._add(name, ConstantParameter(value))

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

    def add_output(self, name: str, to_disk: bool = False,
                   exist_ok: bool = False):
        """Add a new output parameter to the domain.

        Parameters
        ----------
        name : str
            Name of the output parameter.
        to_disk : bool
            Whether to store the output parameter on disk, by default False.
        exist_ok: bool
            Whether to raise an error if the output parameter already exists,
            by default False.

        Example
        -------
        >>> domain = Domain()
        >>> domain.add_output('param1', True)
        >>> domain.space
        {'param1': OutputParameter(to_disk=True)}
        """
        if name in self.output_space:
            if not exist_ok:
                raise KeyError(
                    f"Parameter {name} already exists in the domain! \
                        Choose a different name.")
            return

        self.output_space[name] = Parameter(to_disk)
#                                                                       Getters
# =============================================================================

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
                for _, parameter in self.continuous.input_space.items()]
        )

    def _filter(self, type: Type[Parameter]) -> Domain:
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
            input_space={
                name: parameter for name, parameter in self.input_space.items()
                if isinstance(parameter, type)}
        )

#                                                                 Miscellaneous
# =============================================================================


def make_nd_continuous_domain(bounds: np.ndarray | List[List[float]],
                              dimensionality: Optional[int] = None) -> Domain:
    """Create a continuous domain.

    Parameters
    ----------
    bounds : numpy.ndarray
        A 2D numpy array of shape (dimensionality, 2) specifying the lower
        and upper bounds of every dimension.
    dimensionality : int
        The number of dimensions, optional. If not given, it is inferred
        from the shape of the bounds. Argument is still present for legacy
        reasons.

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
    input_space = {}

    # bounds is a list of lists, convert to numpy array:
    bounds = np.array(bounds)

    dimensionality = bounds.shape[0]

    for dim in range(dimensionality):
        input_space[f"x{dim}"] = ContinuousParameter(
            lower_bound=bounds[dim, 0], upper_bound=bounds[dim, 1])

    return Domain(input_space=input_space)


def _domain_factory(domain: Domain | DictConfig | Path | str) -> Domain:
    # If domain is already a Domain object, return it
    if isinstance(domain, Domain):
        return domain

    # If domain is not given, return an empty Domain object
    elif domain is None:
        return Domain()

    # If domain is a path, load the domain from the file
    elif isinstance(domain, (Path, str)):
        return Domain.from_file(Path(domain))

    # If the domain is a hydra DictConfig, convert it to a Domain object
    elif isinstance(domain, DictConfig):
        return Domain.from_yaml(domain)

    else:
        raise TypeError(
            f"Domain must be of type Domain, DictConfig "
            f"or None, not {type(domain)}")
