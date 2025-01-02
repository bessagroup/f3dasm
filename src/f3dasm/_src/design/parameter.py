"""Parameters for constructing the feasible search space."""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
from pathlib import Path
from typing import Any, ClassVar, Iterable, Optional, Union

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


class StoreProtocol:
    """Base class for storing and loading output data from disk"""
    suffix: int

    def __init__(self, object: Any, path: Path):
        """
        Protocol class for storing and loading output data from disk

        Parameters
        ----------
        object : Any
            object to store
        path : Path
            location to store the object to
        """
        self.path = path
        self.object = object

    def store(self) -> None:
        """
        Protocol class for storing objects to disk

        Raises
        ------
        NotImplementedError
            Raises if the method is not implemented
        """
        raise NotImplementedError()

    def load(self) -> Any:
        """
        Protocol class for loading objects to disk

        Returns
        -------
        Any
            The loaded object

        Raises
        ------
        NotImplementedError
            Raises if the method is not implemented
        """
        raise NotImplementedError()

# =============================================================================


class Parameter:
    """Interface class of a search space parameter."""
    _type: ClassVar[str] = "object"

    def __init__(self, to_disk: bool = False,
                 store_protocol: Optional[StoreProtocol] = None):
        self.to_disk = to_disk
        self.store_protocol = store_protocol

    def __str__(self):
        return f"Parameter(type={self._type}, to_disk={self.to_disk})"

    def __repr__(self):
        return f"{self.__class__.__name__}(to_disk={self.to_disk})"

# =============================================================================


class ConstantParameter(Parameter):
    """Create a search space parameter that is constant.

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

# =============================================================================


class ContinuousParameter(Parameter):
    """
    A search space parameter that is continuous.
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

    def _validate_range(self):
        if self.upper_bound <= self.lower_bound:
            raise ValueError((
                f"The `upper_bound` value must be larger than `lower_bound`. "
                f"(lower_bound={self.lower_bound}, "
                f"upper_bound={self.upper_bound})")
            )

    def to_discrete(self, step: int = 1) -> "DiscreteParameter":
        if step <= 0:
            raise ValueError("The step size must be larger than 0.")
        return DiscreteParameter(
            lower_bound=self.lower_bound,
            upper_bound=self.upper_bound,
            step=step
        )

# =============================================================================


class DiscreteParameter(Parameter):
    """Create a search space parameter that is discrete."""

    def __init__(self, lower_bound=0, upper_bound=1, step=1):
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

    def _validate_range(self):
        if self.upper_bound <= self.lower_bound:
            raise ValueError("Upper bound must be greater than lower bound.")
        if self.step <= 0:
            raise ValueError("Step size must be positive.")

# =============================================================================


class CategoricalParameter(Parameter):
    """Create a search space parameter that is categorical."""
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

    def __eq__(self, other: "CategoricalParameter") -> bool:
        return set(self.categories) == set(other.categories)

    def _check_duplicates(self):
        if len(self.categories) != len(set(self.categories)):
            raise ValueError("Categories contain duplicates!")

# =============================================================================


PARAMETERS = [CategoricalParameter, ConstantParameter,
              ContinuousParameter, DiscreteParameter]
