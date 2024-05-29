import pytest

from f3dasm._src.design.parameter import (_CategoricalParameter,
                                          _ContinuousParameter,
                                          _DiscreteParameter)
from f3dasm.design import Domain


@pytest.fixture(scope="package")
def design():
    # Define the parameters
    parameters = {
        "x1": _ContinuousParameter(lower_bound=2.4, upper_bound=10.3),
        "x2": _DiscreteParameter(lower_bound=5, upper_bound=80),
        "x3": _ContinuousParameter(lower_bound=10.0, upper_bound=380.3),
        "x4": _CategoricalParameter(categories=["test1", "test2", "test3"]),
        "x5": _DiscreteParameter(lower_bound=3, upper_bound=6),
        "x6": _CategoricalParameter(categories=["material1", "material2", "material3"]),
    }

    # Create the design space
    design = Domain(parameters)
    return design


@pytest.fixture(scope="package")
def design2():
    # Define the parameters
    parameters = {
        "x1": _ContinuousParameter(lower_bound=2.4, upper_bound=10.3),
        "x2": _CategoricalParameter(categories=["main"]),
        "x3": _ContinuousParameter(lower_bound=10.0, upper_bound=380.3),
        "x4": _CategoricalParameter(categories=["test" + str(i) for i in range(80)]),
        "x5": _DiscreteParameter(lower_bound=3, upper_bound=6),
        "x6": _CategoricalParameter(categories=["material" + str(i) for i in range(20)]),
    }

    # Create the design space
    design = Domain(parameters)
    return design


@pytest.fixture
def design3():
    # Define the parameters
    parameters = {
        "x1": _ContinuousParameter(lower_bound=2.4, upper_bound=10.3),
        "x2": _DiscreteParameter(lower_bound=5, upper_bound=80),
        "x3": _ContinuousParameter(lower_bound=10.0, upper_bound=380.3),
        "x4": _CategoricalParameter(categories=["test1", "test2", "test3"]),
        "x5": _ContinuousParameter(lower_bound=0.6, upper_bound=7.3),
    }

    # Create the design space
    design = Domain(parameters)
    return design


@pytest.fixture
def design4():
    # Define the parameters
    parameters = {
        "x1": _ContinuousParameter(lower_bound=2.4, upper_bound=10.3),
        "x2": _DiscreteParameter(lower_bound=5, upper_bound=80),
        "x3": _ContinuousParameter(lower_bound=10.0, upper_bound=380.3),
        "x4": _CategoricalParameter(categories=["test1", "test2", "test3"]),
        "x5": _DiscreteParameter(lower_bound=3, upper_bound=6),
    }

    # Create the design space
    design = Domain(parameters)
    return design


@pytest.fixture
def design5():
    # Define the parameters
    parameters = {
        "x1": _ContinuousParameter(lower_bound=2.4, upper_bound=10.3),
        "x2": _DiscreteParameter(lower_bound=5, upper_bound=80),
        "x3": _ContinuousParameter(lower_bound=10.0, upper_bound=380.3),
        "x4": _CategoricalParameter(categories=["test1", "test2", "test3"]),
        "x5": _DiscreteParameter(lower_bound=3, upper_bound=6),
        "x6": _DiscreteParameter(lower_bound=500, upper_bound=532),
    }

    # Create the design space
    design = Domain(parameters)
    return design
