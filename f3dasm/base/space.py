from dataclasses import dataclass, field
from typing import List

import autograd.numpy as np


@dataclass
class ParameterInterface:
    """Interface class of a search space parameter

    Args:
        name (str): name of the parameter
    """

    name: str


@dataclass
class ContinuousParameter(ParameterInterface):
    """Creates a search space parameter that is continuous

    Args:
        lower_bound (float): lower bound of continuous search space
        upper_bound (float): upper bound of continuous search space (exclusive)
    """

    lower_bound: float = field(default=-np.inf)
    upper_bound: float = field(default=np.inf)
    type: str = field(init=False, default="float")

    def __post_init__(self):
        self._check_types()
        self._check_range()

    def _check_types(self) -> None:
        """Check if the boundaries are actually floats"""
        if not isinstance(self.lower_bound, float) or not isinstance(self.upper_bound, float):
            raise TypeError(f"Expect float, got {type(self.lower_bound)} and {type(self.upper_bound)}")

    def _check_range(self) -> None:
        """Check if the lower boundary is lower than the higher boundary"""
        if self.upper_bound < self.lower_bound:
            raise ValueError("not the right range!")

        if self.upper_bound == self.lower_bound:
            raise ValueError("same lower as upper bound!")


@dataclass
class DiscreteParameter(ParameterInterface):
    """Creates a search space parameter that is discrete

    Args:
        lower_bound (int): lower bound of discrete search space
        upper_bound (int): upper bound of discrete search space (exclusive)
    """

    lower_bound: int = field(default=0)
    upper_bound: int = field(default=1)
    type: str = field(init=False, default="int")

    def __post_init__(self):
        self._check_types()
        self._check_range()

    def _check_types(self) -> None:
        """Check if the boundaries are actually ints"""
        if not isinstance(self.lower_bound, int) or not isinstance(self.upper_bound, int):
            raise TypeError(f"Expect integer, got {type(self.lower_bound)} and {type(self.upper_bound)}")

    def _check_range(self) -> None:
        """Check if the lower boundary is lower than the higher boundary"""
        if self.upper_bound < self.lower_bound:
            raise ValueError("not the right range!")

        if self.upper_bound == self.lower_bound:
            raise ValueError("same lower as upper bound!")


@dataclass
class CategoricalParameter(ParameterInterface):
    """Creates a search space parameter that is categorical

    Args:
        categories (list): list of strings that represent available categories
    """

    categories: List[str]
    type: str = field(init=False, default="category")

    def __post_init__(self):
        self._check_types()
        self._check_duplicates()

    def _check_duplicates(self) -> None:
        """Check if there are duplicates in the categories list"""
        if len(self.categories) != len(set(self.categories)):
            raise ValueError("Categories contain duplicates!")

    def _check_types(self) -> None:
        """Check if the entries of the lists are all strings"""

        if not isinstance(self.categories, list):
            raise TypeError(f"Expect list, got {type(self.categories)}")

        for category in self.categories:
            if not isinstance(category, str):
                raise TypeError(f"Expect string, got {type(category)}")


@dataclass
class ConstraintInterface:
    """Interface for constraints"""

    pass
