"""
The Domain is a set of Parameter instances that make up
 the feasible search space.
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
import json
import math
from itertools import zip_longest
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Type

# Third-party core
import numpy as np
from omegaconf import DictConfig, OmegaConf

# Local
from ..errors import DecodeError, EmptyFileError
from .parameter import (CategoricalParameter, CategoricalType,
                        ConstantParameter, ContinuousParameter,
                        DiscreteParameter, LoadFunction, Parameter,
                        StoreFunction)

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
        """The len() method returns the number of input parameters"""
        return len(self.input_space)

    def __eq__(self, __o: Domain) -> bool:
        """Custom equality comparison for Domain objects."""

        if not isinstance(__o, Domain):
            raise TypeError(f"Cannot compare Domain with \
                {type(__o)}")
        return (
            self.input_space == __o.input_space
            and self.output_space == __o.output_space)

    def __bool__(self) -> bool:
        """Check if the Domain object is empty"""
        return bool(self.input_space) or bool(self.output_space)

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
            raise TypeError(f"Cannot add Domain with {type(__o)}")

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

    def _copy(self) -> Domain:
        """
        Return a copy of the Domain object

        Returns
        -------
        Domain
            Copy of the Domain object
        """
        return Domain(
            input_space={k: v._copy() for k, v in self.input_space.items()},
            output_space={k: v._copy() for k, v in self.output_space.items()}
        )

    @property
    def input_names(self) -> List[str]:
        """
        Retrieve the input space names

        Returns
        -------
        List[str]
            List of the names of the input parameters
        """
        return list(self.input_space.keys())

    @property
    def output_names(self) -> List[str]:
        """
        Retrieve the output space names

        Returns
        -------
        List[str]
            List of the names of the output parameters"""
        return list(self.output_space.keys())

    @property
    def continuous(self) -> Domain:
        """Filter the continuous parameters of the domain

        Returns
        -------
        Domain
            Domain object containing the continuous parameters
        """
        return self._filter(ContinuousParameter)

    @property
    def discrete(self) -> Domain:
        """Filter the discrete parameters of the domain

        Returns
        -------
        Domain
            Domain object containing the discrete parameters
        """
        return self._filter(DiscreteParameter)

    @property
    def categorical(self) -> Domain:
        """Filter the categorical parameters of the domain

        Returns
        -------
        Domain
            Domain object containing the categorical parameters
        """
        return self._filter(CategoricalParameter)

    @property
    def constant(self) -> Domain:
        """Filter the constant parameters of the domain

        Returns
        -------
        Domain
            Domain object containing the constant parameters
        """
        return self._filter(ConstantParameter)
#                                                      Alternative constructors
# =============================================================================

    @classmethod
    def from_file(cls: Type[Domain], filename: Path | str) -> Domain:
        """
        Create a Domain object from a JSON file.

        Parameters
        ----------
        filename : Path or str
            Path of the JSON file to load the Domain object from.

        Returns
        -------
        Domain
            Domain object containing the loaded design spaces.

        Examples
        --------
        >>> domain = Domain.from_json('domain.json')
        """
        # convert filename to Path object
        filename = Path(filename).with_suffix('.json')

        # Check if filename exists
        if not filename.exists():
            raise FileNotFoundError(f"Domain file {filename} does not exist.")

        if filename.stat().st_size == 0:
            raise EmptyFileError(filename)

        try:
            with open(filename, 'r') as f:
                domain_dict = json.load(f)
        except json.JSONDecodeError:
            raise DecodeError(filename)

        input_space = {k: Parameter.from_dict(
            v) for k, v in domain_dict['input_space'].items()}
        output_space = {k: Parameter.from_dict(
            v) for k, v in domain_dict['output_space'].items()}

        return cls(input_space=input_space, output_space=output_space)

    @classmethod
    def from_yaml(cls: Type[Domain], cfg: DictConfig) -> Domain:
        """Initialize a Domain from a Hydra YAML configuration file key


        Note
        ----
        The YAML file should have the following structure:

        .. code-block:: yaml

            domain:
                input:
                    <parameter_name>:
                        type: <parameter_type>
                        <parameter_type_specific_parameters>
                    <parameter_name>:
                        type: <parameter_type>
                        <parameter_type_specific_parameters>
                output:
                    <parameter_name>:
                        to_disk: <bool>


        Parameters
        ----------
        cfg : DictConfig
            YAML dictionary key of the domain.

        Returns
        -------
        Domain
            Domain object
        """
        def process_input(items):
            for key, value in items.items():
                _dict = OmegaConf.to_container(value, resolve=True)
                domain.add(name=key, type=_dict.pop('type', None), **_dict)

        def process_output(items):
            for key, value in items.items():
                _dict = OmegaConf.to_container(value, resolve=True)
                domain.add_output(name=key, **_dict)

        domain = cls()

        if 'input' in cfg:
            process_input(cfg.input)
        else:
            process_input(cfg)

        if 'output' in cfg:
            process_output(cfg.output)

        return domain

    @classmethod
    def from_data(cls, input_data: List[Dict[str, Any]],
                  output_data: List[Dict[str, Any]]
                  ) -> Domain:
        """
        Initialize a Domain from input and output data.

        Parameters
        ----------
        input_data : List[Dict[str, Any]]
            List of dictionaries containing the input parameters.
        output_data : List[Dict[str, Any]]
            List of dictionaries containing the output parameters.

        Returns
        -------
        Domain
            Domain object containing the input and output parameter names.
        """
        all_input_parameters, all_output_parameters = set(), set()
        for experiment_input, experiment_output in zip_longest(
                input_data, output_data, fillvalue={}):

            all_input_parameters.update(experiment_input.keys())
            all_output_parameters.update(experiment_output.keys())

        input_names = sorted(list(all_input_parameters))
        output_names = sorted(list(all_output_parameters))

        input_space = {name: Parameter() for name in input_names}
        output_space = {name: Parameter() for name in output_names}

        return cls(input_space=input_space, output_space=output_space)

#                                                                        Export
# =============================================================================

    def store(self, filename: Path | str) -> None:
        """
        Store the Domain object and its parameters as a JSON file.

        Parameters
        ----------
        filename : Path or str
            Path of the JSON file to store the Domain object.

        Examples
        --------
        >>> domain.to_json('domain.json')
        """
        domain_dict = {
            'input_space': {k: v.to_dict()
                            for k, v in self.input_space.items()},
            'output_space': {k: v.to_dict()
                             for k, v in self.output_space.items()}
        }
        with open(Path(filename).with_suffix('.json'), 'w') as f:
            json.dump(domain_dict, f, indent=4)

#                                                  Append and remove parameters
# =============================================================================

    def _add(self, name: str, parameter: Parameter):
        """
        Add a new input parameter to the domain.

        Parameters
        ----------
        name : str
            Name of the input parameter.
        parameter : Parameter
            Parameter object to be added to the domain.
        """
        # Check if parameter is already in the domain
        if name in self.input_space:
            raise KeyError(
                f"Parameter {name} already exists in the domain! \
                     Choose a different name.")

        self.input_space[name] = parameter

    def add_parameter(self, name: str,
                      to_disk=False,
                      store_function: Optional[StoreFunction] = None,
                      load_function: Optional[LoadFunction] = None):
        """Add a new parameter to the domain.

        Parameters
        ----------
        name : str
            Name of the input parameter.
        store_function : StoreFunction, optional
            Function to store the parameter, by default None.
        load_function : LoadFunction, optional
            Function to load the parameter, by default None.

        Example
        -------
        >>> domain = Domain()
        >>> domain.add_parameter('param1', store_function, load_function)
        >>> domain.input_space
        {'param1': Parameter(store_function=store_function,
        load_function=load_function)}
        """
        self._add(name, Parameter(store_function=store_function,
                                  load_function=load_function,
                                  to_disk=to_disk))

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
        >>> domain.input_space
        {'param1': DiscreteParameter(lower_bound=0, upper_bound=10, step=2)}

        Note
        ----
        If the lower and upper bound are equal, then a constant parameter
        will be added to the domain!
        """
        if low == high:
            self.add_constant(name, low)
        else:
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
        >>> domain.input_space
        {'param1': ContinuousParameter(lower_bound=0.,
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
        >>> domain.input_space
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
        >>> domain.input_space
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
        >>> domain.input_space
        {'param1': ContinuousParameter(lower_bound=0., upper_bound=1.)}
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
                   exist_ok: bool = False,
                   store_function: Optional[StoreFunction] = None,
                   load_function: Optional[LoadFunction] = None):
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
        >>> domain.input_space
        {'param1': OutputParameter(to_disk=True)}
        """
        if name in self.output_space:
            if not exist_ok:
                raise KeyError(
                    f"Parameter {name} already exists in the domain! \
                        Choose a different name.")
            return

        self.output_space[name] = Parameter(to_disk=to_disk,
                                            store_function=store_function,
                                            load_function=load_function)
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
        >>> domain.input_space = {
        ...     'param1': ContinuousParameter(lower_bound=0, upper_bound=1),
        ...     'param2': ContinuousParameter(lower_bound=-1, upper_bound=1),
        ...     'param3': ContinuousParameter(lower_bound=0, upper_bound=10)
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
        >>> domain.input_space = {
        ...     'param1': ContinuousParameter(lower_bound=0., upper_bound=1.),
        ...     'param2': DiscreteParameter(lower_bound=0, upper_bound=8),
        ...     'param3': CategoricalParameter(categories=['cat1', 'cat2'])
        ... }
        >>> filtered_domain = domain.filter_parameters(ContinuousParameter)
        >>> filtered_domain.input_space
        {'param1': ContinuousParameter(lower_bound=0, upper_bound=1)}

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
    """
    Factory function to create a Domain object from various input types.

    Parameters
    ----------
    domain : Domain | DictConfig | Path | str
        The domain to be converted to a Domain object.

    Returns
    -------
    Domain
        Domain object
    """
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
