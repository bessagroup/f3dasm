import pytest

from f3dasm.design.design import DesignSpace
from f3dasm.design.parameter import (CategoricalParameter, ContinuousParameter,
                                     DiscreteParameter)


@pytest.fixture(scope="package")
def doe():
    x1 = ContinuousParameter(name="x1", lower_bound=2.4, upper_bound=10.3)
    x2 = DiscreteParameter(name="x2", lower_bound=5, upper_bound=80)
    x3 = ContinuousParameter(name="x3", lower_bound=10.0, upper_bound=380.3)
    x4 = CategoricalParameter(name="x4", categories=["test1", "test2", "test3"])
    x5 = DiscreteParameter(name="x5", lower_bound=2, upper_bound=3)

    y1 = ContinuousParameter(name="y1")
    y2 = ContinuousParameter(name="y2")
    designspace = [x1, x2, x3, x4, x5]
    output_space = [y1, y2]

    doe = DesignSpace(input_space=designspace, output_space=output_space)
    return doe
