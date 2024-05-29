from typing import Callable, Dict

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.strategies import (SearchStrategy, composite, floats, integers,
                                   text)

from f3dasm._src.design.parameter import (_CategoricalParameter,
                                          _ContinuousParameter,
                                          _DiscreteParameter, _Parameter)
from f3dasm.design import Domain

pytestmark = pytest.mark.smoke


@composite
def design_space(draw: Callable[[SearchStrategy[int]], int], min_value: int = 1, max_value: int = 20):
    number_of_input_parameters = draw(integers(min_value, max_value))
    _name = text(alphabet="abcdefghijklmnopqrstuvwxyz",
                 min_size=10, max_size=10)

    def get_space(number_of_parameters: int) -> Dict[str, _Parameter]:
        space = {}
        names = []
        for _ in range(number_of_parameters):
            names.append(_name.filter(lambda x: x not in names))

        for i in range(number_of_parameters):
            parameter: _Parameter = np.random.choice(
                a=["ContinuousSpace", "DiscreteSpace", "CategoricalSpace"]
            )
            name = names[i]

            if parameter == "ContinuousSpace":
                lower_bound, upper_bound = (
                    draw(floats(max_value=0.0)),
                    draw(floats(min_value=0.1)),
                )

                space[name] = _ContinuousParameter(
                    lower_bound=lower_bound, upper_bound=upper_bound)

            elif parameter == "DiscreteSpace":
                lower_bound, upper_bound = (
                    draw(integers(max_value=0)),
                    draw(integers(min_value=1)),
                )

                space[name] = _DiscreteParameter(
                    lower_bound=lower_bound, upper_bound=upper_bound)
            elif parameter == "CategoricalSpace":
                categories = ["test1", "test2"]
                space[name] = _CategoricalParameter(categories=categories)

        return space

    design_space = Domain(
        space=get_space(number_of_input_parameters),
    )
    return design_space


@given(design_space())
@settings(max_examples=10)
def test_check_length_input_when_adding_parameter(design: Domain):
    length_input_space = len(design.space)
    parameter = _DiscreteParameter()
    kwargs = {'low': parameter.lower_bound, 'high': parameter.upper_bound}
    design.add(name="test", type='int', **kwargs)
    assert length_input_space + 1 == (len(design.space))


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
