"""Parameters for constructing the feasible search space."""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional, Sequence, Union

# Third-party
import numpy as np

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================

CategoricalType = Union[None, int, float, str]


@dataclass
class Parameter:
    """Interface class of a search space parameter

    Parameters
    ----------
    """
    _type: ClassVar[str] = field(init=False, default="object")


@dataclass
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

    value: Any
    _type: ClassVar[str] = field(init=False, default="object")

    def __post_init__(self):
        self._check_hashable()

    def _check_hashable(self):
        """Check if the value is hashable."""
        try:
            hash(self.value)
        except TypeError:
            raise TypeError("The value must be hashable.")


@dataclass
class ContinuousParameter(Parameter):
    """
    A search space parameter that is continuous.

    Attributes
    ----------
    lower_bound : float, optional
        The lower bound of the continuous search space. Defaults to negative infinity.
    upper_bound : float, optional
        The upper bound of the continuous search space (exclusive). Defaults to infinity.
    log : bool, optional
        Whether the search space is logarithmic. Defaults to False.

    Raises
    ------
    TypeError
        If the boundaries are not floats.
    ValueError
        If the upper bound is less than the lower bound, or if the lower bound is equal to the upper bound.

    Notes
    -----
    This class inherits from the `Parameter` class and adds the ability to specify a continuous search space.
    """

    lower_bound: float = field(default=-np.inf)
    upper_bound: float = field(default=np.inf)
    log: bool = field(default=False)
    _type: ClassVar[str] = field(init=False, default="float")

    def __post_init__(self):

        if self.log and self.lower_bound <= 0.0:
            raise ValueError(
                f"The `lower_bound` value must be larger than 0 for a log distribution "
                f"(low={self.lower_bound}, high={self.upper_bound})."
            )

        self._check_types()
        self._check_range()

    def _check_types(self):
        """Check if the boundaries are actually floats"""
        if isinstance(self.lower_bound, int):
            self.lower_bound = float(self.lower_bound)

        if isinstance(self.upper_bound, int):
            self.upper_bound = float(self.upper_bound)

        if not isinstance(self.lower_bound, float) or not isinstance(self.upper_bound, float):
            raise TypeError(
                f"Expect float, got {type(self.lower_bound)} and {type(self.upper_bound)}")

    def _check_range(self):
        """Check if the lower boundary is lower than the higher boundary"""
        if self.upper_bound <= self.lower_bound:
            raise ValueError(f"The `upper_bound` value must be larger than the `lower_bound` value "
                             f"(lower_bound={self.lower_bound}, higher_bound={self.upper_bound}")


@dataclass
class DiscreteParameter(Parameter):
    """Create a search space parameter that is discrete

    Parameters
    ----------
    lower_bound : int, optional
        lower bound of discrete search space
    upper_bound : int, optional
        upper bound of discrete search space (exclusive)
    step : int, optional
        step size of discrete search space
    """

    lower_bound: int = field(default=0)
    upper_bound: int = field(default=1)
    step: int = field(default=1)
    _type: ClassVar[str] = field(init=False, default="int")

    def __post_init__(self):

        self._check_types()
        self._check_range()

    def _check_types(self):
        """Check if the boundaries are actually ints"""
        if not isinstance(self.lower_bound, int) or not isinstance(self.upper_bound, int):
            raise TypeError(
                f"Expect integer, got {type(self.lower_bound)} and {type(self.upper_bound)}")

    def _check_range(self):
        """Check if the lower boundary is lower than the higher boundary"""
        if self.upper_bound < self.lower_bound:
            raise ValueError("not the right range!")

        if self.upper_bound == self.lower_bound:
            raise ValueError("same lower as upper bound!")

        if self.step <= 0:
            raise ValueError("step size must be larger than 0!")


@dataclass
class CategoricalParameter(Parameter):
    """Create a search space parameter that is categorical

    Parameters
    ----------
    categories
        list of strings that represent available categories
    """

    categories: Sequence[CategoricalType]
    _type: str = field(init=False, default="category")

    def __post_init__(self):
        self._check_duplicates()

    def _check_duplicates(self):
        """Check if there are duplicates in the categories list"""
        if len(self.categories) != len(set(self.categories)):
            raise ValueError("Categories contain duplicates!")


PARAMETERS = [CategoricalParameter, ConstantParameter, ContinuousParameter, DiscreteParameter]


def create_inputvariable(type: str, lower_bound: Optional[int | float] = None,
                         upper_bound: Optional[int | float] = None,
                         values: Optional[CategoricalType | Sequence[CategoricalType]] = None) -> Parameter:
    if type.lower == "float":
        return ContinuousParameter(lower_bound=lower_bound, upper_bound=upper_bound)

    elif type.upper == "int":
        return DiscreteParameter(lower_bound=lower_bound, upper_bound=upper_bound)

    elif type.lower == "category":
        return CategoricalParameter(categories=values)

    elif type.lower == "constant":
        return ConstantParameter(value=values)

    else:
        raise ValueError(f"Unknown type argument: {type}. Choose from 'float', 'int', 'category' or 'constant'.")
