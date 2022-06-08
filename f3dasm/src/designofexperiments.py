# %%
from dataclasses import dataclass, field
from typing import List

from f3dasm.src.space import ConstraintInterface, SpaceInterface


@dataclass
class DesignOfExperiments:
    """Design of Experiments"""

    pass


@dataclass
class DoE:

    space: List[SpaceInterface] = field(default_factory=list)
    constraints: List[ConstraintInterface] = field(default_factory=list)

    def add_space(self, space: SpaceInterface) -> None:
        self.space.append(space)
        return

    def add_constraint(self, constraint: ConstraintInterface) -> None:
        self.constraints.append(constraint)
        return
