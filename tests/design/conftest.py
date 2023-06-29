import pytest

from f3dasm.design.design import DesignSpace
from f3dasm.design.parameter import (CategoricalParameter, ContinuousParameter,
                                     DiscreteParameter)


@pytest.fixture(scope="package")
def doe():
    x1 = ContinuousParameter(lower_bound=2.4, upper_bound=10.3)
    x2 = DiscreteParameter(lower_bound=5, upper_bound=80)
    x3 = ContinuousParameter(lower_bound=10.0, upper_bound=380.3)
    x4 = CategoricalParameter(categories=["test1", "test2", "test3"])
    x5 = DiscreteParameter(lower_bound=2, upper_bound=3)

    y1 = ContinuousParameter()
    y2 = ContinuousParameter()
    designspace = {'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5}
    output_space = {'y1': y1, 'y2': y2}

    doe = DesignSpace(input_space=designspace, output_space=output_space)
    return doe


@pytest.fixture(scope="package")
def design_space():

    input_space = {
        'x1': ContinuousParameter(-5.12, 5.12),
        'x2': DiscreteParameter(-3, 3),
        'x3': CategoricalParameter(["red", "green", "blue"])
    }

    output_space = {
        'y': ContinuousParameter(-10.0, 10.0)
    }

    return DesignSpace(input_space=input_space, output_space=output_space)


@pytest.fixture(scope="package")
def continuous_parameter():
    lower_bound = 3.3
    upper_bound = 3.8
    return ContinuousParameter(lower_bound=lower_bound, upper_bound=upper_bound)


@pytest.fixture(scope="package")
def discrete_parameter():
    lower_bound = 3
    upper_bound = 6
    return DiscreteParameter(lower_bound=lower_bound, upper_bound=upper_bound)


@pytest.fixture(scope="package")
def categorical_parameter():
    categories = ["test1", "test2", "test3"]
    return CategoricalParameter(categories=categories)
