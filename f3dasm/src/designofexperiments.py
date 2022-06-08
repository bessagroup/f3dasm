
# %%
from abc import ABC
from dataclasses import dataclass

@dataclass
class ContinuousParameter:
    name: str
    lower_bound: float
    upper_bound: float

    def __post_init__(self):
        self.check_types()
        self.check_range()

    def check_types(self):
        if not isinstance(self.lower_bound, float) or not isinstance(self.upper_bound, float):
            raise ValueError(f"Expect float, got {type(self.lower_bound)} and {type(self.upper_bound)}")

    def check_range(self):
        if self.upper_bound < self.lower_bound:
            raise ValueError("not the right range!")


@dataclass
class DesignOfExperiments:
    """Design of Experiments"""
    pass