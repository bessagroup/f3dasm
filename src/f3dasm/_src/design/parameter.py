"""Parameters for constructing the feasible search space."""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
from typing import Any, ClassVar, Iterable, Optional, Protocol, Union

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

CategoricalType = Union[None, int, float, str]

# =============================================================================


class StoreFunction(Protocol):
    """Base class for storing and loading output data from disk"""

    def __call__(object: Any, path: str) -> str:
        """
        Protocol class for storing output data from disk.

        Parameters
        ----------
        object : Any
            The object to store.
        path : str
            The location to store the object to.

        Notes
        -----
        The function should store the object to the location specified by the
        path parameter. The suffix of the file should be determined by the
        object type, and is not yet implemented in the path!
        """
        ...


class LoadFunction(Protocol):
    """Base class for storing and loading output data from disk"""

    def __call__(path: str) -> Any:
        """
        Protocol class for loading output data from disk.

        Parameters
        ----------
        path : str
            The location to load the object from.

        Returns
        -------
        Any
            The loaded object.
        """
        ...


# =============================================================================


class Parameter:
    """Interface class of a search space parameter."""
    _type: ClassVar[str] = "object"

    def __init__(self, to_disk: bool = False,
                 store_function: Optional[StoreFunction] = None,
                 load_function: Optional[LoadFunction] = None):
        """
        Initialize the Parameter.

        Parameters
        ----------
        to_disk : bool, optional
            Whether the parameter should be saved to disk. Defaults to False.
        store_function : Optional[StoreFunction], optional
            Function to store the parameter to disk. Defaults to None.
        load_function : Optional[LoadFunction], optional
            Function to load the parameter from disk. Defaults to None.

        Raises
        ------
        ValueError
            If `to_disk` is False but either `store_function` or
            `load_function` is not None.

        Examples
        --------
        >>> param = Parameter(to_disk=True)
        >>> print(param)
        Parameter(type=object, to_disk=True)
        """

        if not to_disk and (
                store_function is not None or load_function is not None):
            raise ValueError(("If 'to_disk' is False, 'store_function' and"
                              "load_function' must be None.")
                             )

        self.to_disk = to_disk
        self.store_function = store_function
        self.load_function = load_function

    def __str__(self):
        return f"Parameter(type={self._type}, to_disk={self.to_disk})"

    def __repr__(self):
        return f"{self.__class__.__name__}(to_disk={self.to_disk})"

    def __eq__(self, __o: Parameter):
        return self.to_disk == __o.to_disk

    def __add__(self, __o: Parameter) -> Parameter:
        return self

# =============================================================================


class ConstantParameter(Parameter):
    """
    Create a search space parameter that is constant.

    Parameters
    ----------
    value : Any
        The constant value of the parameter.

    Attributes
    ----------
    _type : str
        The type of the parameter, which is always 'object'.

    Raises
    ------
    TypeError
        If the value is not hashable.

    Examples
    --------
    >>> param = ConstantParameter(value=5)
    >>> print(param)
    ConstantParameter(value=5)
    """

    def __init__(self, value: Any):
        super().__init__()
        self.value = value
        self._validate_hashable()

    def __add__(self, other: Parameter):
        if isinstance(other, ConstantParameter):
            if self.value == other.value:
                return self
            else:
                return CategoricalParameter(
                    categories=[self.value, other.value])

        if isinstance(other, CategoricalParameter):
            return self.to_categorical() + other

        if isinstance(other, DiscreteParameter):
            return self.to_categorical() + other

        if isinstance(other, ContinuousParameter):
            raise ValueError("Cannot add continuous parameter to constant!")

    def to_categorical(self) -> "CategoricalParameter":
        return CategoricalParameter(categories=[self.value])

    def _validate_hashable(self):
        """Check if the value is hashable."""
        try:
            hash(self.value)
        except TypeError:
            raise TypeError("The value must be hashable.")

    def __str__(self):
        return f"ConstantParameter(value={self.value})"

    def __repr__(self):
        return f"{self.__class__.__name__}(value={repr(self.value)})"

    def __eq__(self, __o: Parameter) -> bool:
        if not isinstance(__o, ConstantParameter):
            return False

        return self.value == __o.value

# =============================================================================


class ContinuousParameter(Parameter):
    """
    A search space parameter that is continuous.

    Parameters
    ----------
    lower_bound : float, optional
        The lower bound of the parameter. Defaults to -inf.
    upper_bound : float, optional
        The upper bound of the parameter. Defaults to inf.
    log : bool, optional
        Whether the parameter should be on a log scale. Defaults to False.

    Raises
    ------
    ValueError
        If `log` is True and `lower_bound` is less than or equal to 0.
        If `upper_bound` is less than or equal to `lower_bound`.

    Examples
    --------
    >>> param = ContinuousParameter(lower_bound=0.0, upper_bound=1.0)
    >>> print(param)
    ContinuousParameter(lower_bound=0.0, upper_bound=1.0, log=False)
    """
    _type: ClassVar[str] = "float"

    def __init__(self, lower_bound: float = float('-inf'),
                 upper_bound: float = float('inf'), log: bool = False):
        super().__init__()
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)
        self.log = log

        if self.log and self.lower_bound <= 0.0:
            raise ValueError((
                f"The `lower_bound` value must be larger than 0 for a "
                f"log distribution (low={self.lower_bound}, "
                f"high={self.upper_bound})."
            ))
        self._validate_range()

    def __add__(self, other: Parameter) -> "ContinuousParameter":
        if not isinstance(other, ContinuousParameter):
            raise ValueError(
                "Cannot add non-continuous parameter to continuous!")
        if self.log != other.log:
            raise ValueError(
                "Cannot add continuous parameters with different log scales!")
        if self.lower_bound > other.upper_bound or \
                other.lower_bound > self.upper_bound:
            raise ValueError("Ranges do not coincide, cannot add")

        return ContinuousParameter(
            lower_bound=min(self.lower_bound, other.lower_bound),
            upper_bound=max(self.upper_bound, other.upper_bound)
        )

    def __str__(self):
        return (f"ContinuousParameter(lower_bound={self.lower_bound}, "
                f"upper_bound={self.upper_bound}, log={self.log})")

    def __repr__(self):
        return (f"{self.__class__.__name__}(lower_bound={self.lower_bound}, "
                f"upper_bound={self.upper_bound}, log={self.log})")

    def __eq__(self, __o: Parameter) -> bool:
        if not isinstance(__o, ContinuousParameter):
            return False

        return (self.lower_bound == __o.lower_bound and self.upper_bound
                == __o.upper_bound and self.log == __o.log)

    def _validate_range(self):
        if self.upper_bound <= self.lower_bound:
            raise ValueError((
                f"The `upper_bound` value must be larger than `lower_bound`. "
                f"(lower_bound={self.lower_bound}, "
                f"upper_bound={self.upper_bound})")
            )

    def to_discrete(self, step: int = 1) -> "DiscreteParameter":
        """
        Convert the continuous parameter to a discrete parameter.

        Parameters
        ----------
        step : int, optional
            The step size for the discrete parameter. Defaults to 1.

        Returns
        -------
        DiscreteParameter
            The converted discrete parameter.

        Raises
        ------
        ValueError
            If the step size is less than or equal to 0.

        Examples
        --------
        >>> param = ContinuousParameter(lower_bound=0.0, upper_bound=1.0)
        >>> discrete_param = param.to_discrete(step=0.1)
        >>> print(discrete_param)
        DiscreteParameter(lower_bound=0.0, upper_bound=1.0, step=0.1)
        """
        if step <= 0:
            raise ValueError("The step size must be larger than 0.")
        return DiscreteParameter(
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            step=step
        )

# =============================================================================


class DiscreteParameter(Parameter):
    """
    Create a search space parameter that is discrete.

    Parameters
    ----------
    lower_bound : int, optional
        The lower bound of the parameter. Defaults to 0.
    upper_bound : int, optional
        The upper bound of the parameter. Defaults to 1.
    step : int, optional
        The step size for the parameter. Defaults to 1.

    Raises
    ------
    ValueError
        If `upper_bound` is less than or equal to `lower_bound`.
        If `step` is less than or equal to 0.

    Examples
    --------
    >>> param = DiscreteParameter(lower_bound=0, upper_bound=10, step=1)
    >>> print(param)
    DiscreteParameter(lower_bound=0, upper_bound=10, step=1)
    """

    def __init__(self, lower_bound: int = 0,
                 upper_bound: int = 1, step: int = 1):
        super().__init__()
        self.lower_bound = int(lower_bound)
        self.upper_bound = int(upper_bound)
        self.step = step
        self._type = "int"

        self._validate_range()

    def __str__(self):
        return (f"DiscreteParameter(lower_bound={self.lower_bound}, "
                f"upper_bound={self.upper_bound}, step={self.step})")

    def __repr__(self):
        return (f"{self.__class__.__name__}(lower_bound={self.lower_bound}, "
                f"upper_bound={self.upper_bound}, step={self.step})")

    def __add__(self, other: Parameter) -> "DiscreteParameter":
        if isinstance(other, CategoricalParameter):
            return other + self
        if isinstance(other, ConstantParameter):
            return other.to_categorical() + self
        if isinstance(other, ContinuousParameter):
            raise ValueError("Cannot add continuous parameter to discrete!")
        return self  # Assuming the same discrete parameters are being added.

    def __eq__(self, __o: Parameter) -> bool:
        if not isinstance(__o, DiscreteParameter):
            return False

        return (self.lower_bound == __o.lower_bound and self.upper_bound
                == __o.upper_bound and self.step
                == __o.step)

    def _validate_range(self):
        if self.upper_bound <= self.lower_bound:
            raise ValueError("Upper bound must be greater than lower bound.")
        if self.step <= 0:
            raise ValueError("Step size must be positive.")

# =============================================================================


class CategoricalParameter(Parameter):
    """
    Create a search space parameter that is categorical.

    Parameters
    ----------
    categories : Iterable[Any]
        The categories of the parameter.

    Raises
    ------
    ValueError
        If the categories contain duplicates.

    Examples
    --------
    >>> param = CategoricalParameter(categories=['a', 'b', 'c'])
    >>> print(param)
    CategoricalParameter(categories=['a', 'b', 'c'])
    """
    _type: ClassVar[str] = "object"

    def __init__(self, categories: Iterable[Any]):
        super().__init__()
        self.categories = categories
        self._check_duplicates()

    def __str__(self):
        return f"CategoricalParameter(categories={self.categories})"

    def __repr__(self):
        return (f"{self.__class__.__name__}"
                f"(categories={list(self.categories)})")

    def __add__(self, other: Parameter) -> "CategoricalParameter":
        if isinstance(other, CategoricalParameter):
            joint_categories = list(set(self.categories + other.categories))
        elif isinstance(other, ConstantParameter):
            joint_categories = list(set(self.categories + [other.value]))
        elif isinstance(other, DiscreteParameter):
            joint_categories = list(set(self.categories + list(range(
                other.lower_bound, other.upper_bound, other.step))))
        elif isinstance(other, ContinuousParameter):
            raise ValueError("Cannot add continuous parameter to categorical!")
        else:
            raise ValueError(
                f"Cannot add parameter of type {type(other)} to categorical.")
        return CategoricalParameter(joint_categories)

    def __eq__(self, other: CategoricalParameter) -> bool:
        return set(self.categories) == set(other.categories)

    def _check_duplicates(self):
        if len(self.categories) != len(set(self.categories)):
            raise ValueError("Categories contain duplicates!")

# =============================================================================


PARAMETERS = [CategoricalParameter, ConstantParameter,
              ContinuousParameter, DiscreteParameter]
