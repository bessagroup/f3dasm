"""Parameters for constructing the feasible search space."""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
from dataclasses import dataclass, field
from typing import Any, ClassVar, Sequence, Union

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
class _Parameter:
    """Interface class of a search space parameter

    Parameters
    ----------
    """
    _type: ClassVar[str] = field(init=False, default="object")


@dataclass
class _OutputParameter(_Parameter):
    to_disk: bool = field(default=False)


@dataclass
class _ConstantParameter(_Parameter):
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

    def __add__(self, __o: _Parameter
                ) -> _ConstantParameter | _CategoricalParameter:
        if isinstance(__o, _ConstantParameter):
            if self.value == __o.value:
                return self
            else:
                return _CategoricalParameter(
                    categories=[self.value, __o.value])

        if isinstance(__o, _CategoricalParameter):
            return self.to_categorical() + __o

        if isinstance(__o, _DiscreteParameter):
            return self.to_categorical() + __o

        if isinstance(__o, _ContinuousParameter):
            raise ValueError("Cannot add continuous parameter to constant!")

    def to_categorical(self) -> _CategoricalParameter:
        return _CategoricalParameter(categories=[self.value])

    def _check_hashable(self):
        """Check if the value is hashable."""
        try:
            hash(self.value)
        except TypeError:
            raise TypeError("The value must be hashable.")


@dataclass
class _ContinuousParameter(_Parameter):
    """
    A search space parameter that is continuous.

    Attributes
    ----------
    lower_bound : float, optional
        The lower bound of the continuous search space.
         Defaults to negative infinity.
    upper_bound : float, optional
        The upper bound of the continuous search space (exclusive).
         Defaults to infinity.
    log : bool, optional
        Whether the search space is logarithmic. Defaults to False.

    Raises
    ------
    TypeError
        If the boundaries are not floats.
    ValueError
        If the upper bound is less than the lower bound, or if the
         lower bound is equal to the upper bound.

    Note
    ----
    This class inherits from the `Parameter` class and adds the ability
     to specify a continuous search space.
    """

    lower_bound: float = field(default=-np.inf)
    upper_bound: float = field(default=np.inf)
    log: bool = field(default=False)
    _type: ClassVar[str] = field(init=False, default="float")

    def __post_init__(self):

        if self.log and self.lower_bound <= 0.0:
            raise ValueError(
                f"The `lower_bound` value must be larger than 0 for a \
                     log distribution "
                f"(low={self.lower_bound}, high={self.upper_bound})."
            )

        self._check_types()
        self._check_range()

    def __add__(self, __o: _Parameter) -> _ContinuousParameter:
        if not isinstance(__o, _ContinuousParameter):
            raise ValueError(
                "Cannot add non-continuous parameter to continuous!")

        if self.log != __o.log:
            raise ValueError(
                "Cannot add continuous parameters with different log scales!")

        if self.lower_bound == __o.lower_bound \
                and self.upper_bound == __o.upper_bound:
            # If both lower and upper bounds are the same,
            # return the first object
            return self

        if self.lower_bound > __o.upper_bound \
                or __o.lower_bound > self.upper_bound:
            # If the ranges do not coincide, raise ValueError
            raise ValueError("Ranges do not coincide, cannot add")

        # For other scenarios, join the ranges
        return _ContinuousParameter(
            lower_bound=min(self.lower_bound, __o.lower_bound),
            upper_bound=max(self.upper_bound, __o.upper_bound))

    def _check_types(self):
        """Check if the boundaries are actually floats"""
        if isinstance(self.lower_bound, int):
            self.lower_bound = float(self.lower_bound)

        if isinstance(self.upper_bound, int):
            self.upper_bound = float(self.upper_bound)

        if not isinstance(
                self.lower_bound, float) or not isinstance(
                    self.upper_bound, float):
            raise TypeError(
                f"Expect float, got {type(self.lower_bound).__name__} \
                 and {type(self.upper_bound).__name__}")

    def _check_range(self):
        """Check if the lower boundary is lower than the higher boundary"""
        if self.upper_bound <= self.lower_bound:
            raise ValueError(f"The `upper_bound` value must be larger than \
                             the `lower_bound` value "
                             f"(lower_bound={self.lower_bound}, \
                                 higher_bound={self.upper_bound}")

    def to_discrete(self, step: int = 1) -> _DiscreteParameter:
        """Convert the continuous parameter to a discrete parameter.

        Parameters
        ----------
        step : int
            The step size of the discrete search space, which defaults to 1.

        Returns
        -------
        DiscreteParameter
            The discrete parameter.

        Raises
        ------
        ValueError
            If the step size is less than or equal to 0.

        """
        if step <= 0:
            raise ValueError("The step size must be larger than 0.")

        return _DiscreteParameter(
            lower_bound=int(self.lower_bound),
            upper_bound=int(self.upper_bound),
            step=step
        )


@dataclass
class _DiscreteParameter(_Parameter):
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

    def __add__(self, __o: _Parameter) -> _DiscreteParameter:
        if isinstance(__o, _DiscreteParameter):
            if self.lower_bound == __o.lower_bound and \
                    self.upper_bound == __o.upper_bound and \
                    self.step == __o.step:
                return self

        if isinstance(__o, _CategoricalParameter):
            return __o + self

        if isinstance(__o, _ConstantParameter):
            return __o.to_categorical() + self

        if isinstance(__o, _ContinuousParameter):
            raise ValueError("Cannot add continuous parameter to discrete!")

    def _check_types(self):
        """Check if the boundaries are actually ints"""
        if not isinstance(self.lower_bound, int) or not isinstance(
                self.upper_bound, int):
            raise TypeError(
                f"Expect integer, got {type(self.lower_bound).__name__} and \
                     {type(self.upper_bound).__name__}")

    def _check_range(self):
        """Check if the lower boundary is lower than the higher boundary"""
        if self.upper_bound < self.lower_bound:
            raise ValueError("not the right range!")

        if self.upper_bound == self.lower_bound:
            raise ValueError("same lower as upper bound!")

        if self.step <= 0:
            raise ValueError("step size must be larger than 0!")


@dataclass
class _CategoricalParameter(_Parameter):
    """Create a search space parameter that is categorical

    Parameters
    ----------
    categories
        list of strings that represent available categories
    """

    categories: Sequence[CategoricalType]
    _type: str = field(init=False, default="object")

    def __post_init__(self):
        self._check_duplicates()

    def __add__(self, __o: _Parameter) -> _CategoricalParameter:
        if isinstance(__o, _CategoricalParameter):
            # join unique categories
            joint_categories = list(set(self.categories + __o.categories))

        if isinstance(__o, _ConstantParameter):
            joint_categories = list(set(self.categories + [__o.value]))

        if isinstance(__o, _DiscreteParameter):
            roll_out_discrete = list(range(
                __o.lower_bound, __o.upper_bound, __o.step))
            joint_categories = list(set(self.categories + roll_out_discrete))

        if isinstance(__o, _ContinuousParameter):
            raise ValueError("Cannot add continuous parameter to categorical!")

        return _CategoricalParameter(joint_categories)

    def __eq__(self, __o: _CategoricalParameter) -> bool:
        return set(self.categories) == set(__o.categories)

    def _check_duplicates(self):
        """Check if there are duplicates in the categories list"""
        if len(self.categories) != len(set(self.categories)):
            raise ValueError("Categories contain duplicates!")


PARAMETERS = [_CategoricalParameter, _ConstantParameter,
              _ContinuousParameter, _DiscreteParameter]
