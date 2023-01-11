from typing import Callable, List

import numpy as np
import pytest

pytestmark = pytest.mark.smoke

from hypothesis import given, settings
from hypothesis.strategies import SearchStrategy, composite, floats, integers, text

from f3dasm.base.design import DesignSpace
from f3dasm.base.space import CategoricalParameter, ContinuousParameter, DiscreteParameter, ParameterInterface


@composite
def design_space(draw: Callable[[SearchStrategy[int]], int], min_value: int = 1, max_value: int = 20):
    number_of_input_parameters = draw(integers(min_value, max_value))
    number_of_output_parameters = draw(integers(min_value, max_value))
    _name = text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=10, max_size=10)

    def get_space(number_of_parameters: int) -> List[ParameterInterface]:
        space = []
        names = []
        for _ in range(number_of_parameters):
            names.append(_name.filter(lambda x: x not in names))

        for i in range(number_of_parameters):
            parameter: ParameterInterface = np.random.choice(
                a=["ContinuousSpace", "DiscreteSpace", "CategoricalSpace"]
            )
            name = names[i]

            if parameter == "ContinuousSpace":
                lower_bound, upper_bound = (
                    draw(floats(max_value=0.0)),
                    draw(floats(min_value=0.1)),
                )

                space.append(ContinuousParameter(name=name, lower_bound=lower_bound, upper_bound=upper_bound))

            elif parameter == "DiscreteSpace":
                lower_bound, upper_bound = (
                    draw(integers(max_value=0)),
                    draw(integers(min_value=1)),
                )

                space.append(DiscreteParameter(name=name, lower_bound=lower_bound, upper_bound=upper_bound))
            elif parameter == "CategoricalSpace":
                categories = ["test1", "test2"]
                space.append(CategoricalParameter(name=name, categories=categories))

        return space

    design_space = DesignSpace(
        input_space=get_space(number_of_input_parameters),
        output_space=get_space(number_of_output_parameters),
    )
    return design_space


@given(design_space())
@settings(max_examples=10)
def test_check_length_input_when_adding_parameter(design: DesignSpace):
    length_input_space = len(design.input_space)
    parameter = DiscreteParameter(name="test")
    design.add_input_space(space=parameter)
    assert length_input_space + 1 == (len(design.input_space))


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
