#                                                                       Modules
# =============================================================================

# Standard
import json
from dataclasses import dataclass, field
from typing import Any, ClassVar, List, Type

# Third-party core
import numpy as np

#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)'
__credits__ = ['Martin van der Schelling']
__status__ = 'Stable'
# =============================================================================
#
# =============================================================================


@dataclass
class Parameter:
    """Interface class of a search space parameter

    Parameters
    ----------
    """
    _type: ClassVar[str] = field(init=False, default="object")

    @classmethod
    def get_name(self) -> str:
        """Return the name of the parameter class"""
        return self.__name__


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
    _type: ClassVar[str] = field(init=False, default="float")

    def __post_init__(self):
        self._check_types()
        self._check_range()

    def _check_types(self):
        """Check if the boundaries are actually floats"""
        if not isinstance(self.lower_bound, float) or not isinstance(self.upper_bound, float):
            raise TypeError(
                f"Expect float, got {type(self.lower_bound)} and {type(self.upper_bound)}")

    def _check_range(self):
        """Check if the lower boundary is lower than the higher boundary"""
        if self.upper_bound < self.lower_bound:
            raise ValueError("not the right range!")

        if self.upper_bound == self.lower_bound:
            raise ValueError("same lower as upper bound!")


@dataclass
class DiscreteParameter(Parameter):
    """Create a search space parameter that is discrete

    Parameters
    ----------
    lower_bound
        lower bound of discrete search space
    upper_bound
        upper bound of discrete search space (exclusive)
    values
        list of values defining the discrete search space
    """

    lower_bound: int = field(default=None)
    upper_bound: int = field(default=None)
    values: List[float] = field(default=None)

    _type: ClassVar[str] = field(init=False, default="int")

    def __post_init__(self):
        self._check_values()
        self._check_types()
        self._check_range()

    def _check_values(self):
        """Check that bounds or values are provided."""
        if (self.lower_bound is None or self.upper_bound is None) and self.values is None:
            raise ValueError("Expect at least integer bounds or a list of values.")

        if (self.lower_bound is not None or self.upper_bound is not None) and self.values is not None:
            raise ValueError("Expect integer bounds or a list of values, not both.")

    def _check_types(self):
        """Check if the boundaries are actually ints or if the values are ints or floats"""
        if self.lower_bound and self.upper_bound:
            if (
                not isinstance(self.lower_bound, int)
                or not isinstance(self.upper_bound, int)
            ):
                raise TypeError(
                    f"Expect integer, got {type(self.lower_bound)} and {type(self.upper_bound)}")

        elif self.values:
            if not all([isinstance(v, int) or isinstance(v, float) for v in self.values]):
                raise TypeError(
                    f"Expect a list of ints or floats, got {[type(v) for v in self.values]}.")
            # If there is at least one float, all values are cast to floats
            if any([isinstance(v, float) for v in self.values]):
                self.values = [float(v) for v in self.values]
                self._type = "float"

    def _check_range(self):
        """Check if the lower boundary is lower than the higher boundary"""
        if self.lower_bound and self.upper_bound:
            if self.upper_bound < self.lower_bound:
                raise ValueError("not the right range!")

            if self.upper_bound == self.lower_bound:
                raise ValueError("same lower as upper bound!")

        elif self.values:
            """Check if the lower boundary is lower than the higher boundary"""
            if not all(self.values[i] < self.values[i + 1] for i in range(len(self.values)-1)):
                raise ValueError("Values are not sorted or contain duplicates.")


@dataclass
class CategoricalParameter(Parameter):
    """Create a search space parameter that is categorical

    Parameters
    ----------
    categories
        list of strings that represent available categories
    """

    categories: List[str]
    _type: str = field(init=False, default="category")

    def __post_init__(self):
        self._check_types()
        self._check_duplicates()

    def _check_duplicates(self):
        """Check if there are duplicates in the categories list"""
        if len(self.categories) != len(set(self.categories)):
            raise ValueError("Categories contain duplicates!")

    def _check_types(self):
        """Check if the entries of the lists are all strings"""

        self.categories = list(self.categories)  # Convert to list because hydra parses omegaconf.ListConfig

        for category in self.categories:
            if not isinstance(category, str):
                raise TypeError(f"Expect string, got {type(category)}")


PARAMETERS = [CategoricalParameter, ConstantParameter, ContinuousParameter, DiscreteParameter]
