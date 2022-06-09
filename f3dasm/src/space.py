from dataclasses import dataclass, field
from typing import List


@dataclass
class SpaceInterface:
    name: str


@dataclass
class ContinuousSpace(SpaceInterface):
    """Define a continuous parameter for your search space.

    Args:
        lower_bound (float): _description_
        upper_bound (float): _description_

    """

    lower_bound: float = field(default=0.0)
    upper_bound: float = field(default=1.0)

    def __post_init__(self):
        self.check_types()
        self.check_range()

    def check_types(self) -> None:
        """_summary_

        Raises:
            ValueError: _description_
        """
        if not isinstance(self.lower_bound, float) or not isinstance(
            self.upper_bound, float
        ):
            raise TypeError(
                f"Expect float, got {type(self.lower_bound)} and {type(self.upper_bound)}"
            )

    def check_range(self) -> None:
        """_summary_

        Raises:
            ValueError: _description_
        """
        if self.upper_bound < self.lower_bound:
            raise ValueError("not the right range!")


@dataclass
class DiscreteSpace(SpaceInterface):
    """_summary_

    Args:
        SpaceInterface (_type_): _description_

    Raises:
        ValueError: _description_
        ValueError: _description_
    """

    lower_bound: float = field(default=0)
    upper_bound: float = field(default=1)

    def __post_init__(self):
        self.check_types()
        self.check_range()

    def check_types(self) -> None:
        if not isinstance(self.lower_bound, int) or not isinstance(
            self.upper_bound, int
        ):
            raise TypeError(
                f"Expect integer, got {type(self.lower_bound)} and {type(self.upper_bound)}"
            )

    def check_range(self) -> None:
        if self.upper_bound < self.lower_bound:
            raise ValueError("not the right range!")


@dataclass
class CategoricalSpace(SpaceInterface):
    categories: List[str]

    def __post_init__(self):
        self.check_types()

    def check_types(self) -> None:
        for category in self.categories:
            if not isinstance(category, str):
                raise TypeError(f"Expect string, got {type(category)}")


@dataclass
class ConstraintInterface:
    pass
