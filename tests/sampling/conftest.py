import pytest

from f3dasm.design.design import DesignSpace
from f3dasm.design.parameter import (CategoricalParameter, ContinuousParameter,
                                     DiscreteParameter)
from f3dasm.sampling import LatinHypercube, RandomUniform, SobolSequence


@pytest.fixture(scope="package")
def design():
    # Define the parameters
    x1 = ContinuousParameter(name="x1", lower_bound=2.4, upper_bound=10.3)
    x2 = DiscreteParameter(name="x2", lower_bound=5, upper_bound=80)
    x3 = ContinuousParameter(name="x3", lower_bound=10.0, upper_bound=380.3)
    x4 = CategoricalParameter(name="x4", categories=["test1", "test2", "test3"])
    x5 = DiscreteParameter(name="x5", lower_bound=3, upper_bound=6)
    x6 = CategoricalParameter(name="x6", categories=["material1", "material2", "material3"])

    # Create the design space
    space = [x1, x2, x3, x4, x5, x6]
    design = DesignSpace(space)
    return design


@pytest.fixture(scope="package")
def design2():
    # Define the parameters
    x1 = ContinuousParameter(name="x1", lower_bound=2.4, upper_bound=10.3)
    x2 = CategoricalParameter(name="x2", categories=["main"])
    x3 = ContinuousParameter(name="x3", lower_bound=10.0, upper_bound=380.3)
    x4 = CategoricalParameter(name="x4", categories=["test" + str(i) for i in range(80)])
    x5 = DiscreteParameter(name="x5", lower_bound=3, upper_bound=6)
    x6 = CategoricalParameter(name="x6", categories=["material" + str(i) for i in range(20)])

    # Create the design space
    space = [x1, x2, x3, x4, x5, x6]
    design = DesignSpace(space)
    return design


@pytest.fixture
def design3():
    # Define the parameters
    x1 = ContinuousParameter(name="x1", lower_bound=2.4, upper_bound=10.3)
    x2 = DiscreteParameter(name="x2", lower_bound=5, upper_bound=80)
    x3 = ContinuousParameter(name="x3", lower_bound=10.0, upper_bound=380.3)
    x4 = CategoricalParameter(name="x4", categories=["test1", "test2", "test3"])
    x5 = ContinuousParameter(name="x5", lower_bound=0.6, upper_bound=7.3)

    # Create the design space
    space = [x1, x2, x3, x4, x5]
    design = DesignSpace(space)
    return design


@pytest.fixture
def design4():
    # Define the parameters
    x1 = ContinuousParameter(name="x1", lower_bound=2.4, upper_bound=10.3)
    x2 = DiscreteParameter(name="x2", lower_bound=5, upper_bound=80)
    x3 = ContinuousParameter(name="x3", lower_bound=10.0, upper_bound=380.3)
    x4 = CategoricalParameter(name="x4", categories=["test1", "test2", "test3"])
    x5 = DiscreteParameter(name="x5", lower_bound=3, upper_bound=6)

    # Create the design space
    space = [x1, x2, x3, x4, x5]
    design = DesignSpace(space)
    return design


@pytest.fixture
def design5():
    # Define the parameters
    x1 = ContinuousParameter(name="x1", lower_bound=2.4, upper_bound=10.3)
    x2 = DiscreteParameter(name="x2", lower_bound=5, upper_bound=80)
    x3 = ContinuousParameter(name="x3", lower_bound=10.0, upper_bound=380.3)
    x4 = CategoricalParameter(name="x4", categories=["test1", "test2", "test3"])
    x5 = DiscreteParameter(name="x5", lower_bound=3, upper_bound=6)
    x6 = DiscreteParameter(name="x6", lower_bound=500, upper_bound=532)

    # Create the design space
    space = [x1, x2, x3, x4, x5, x6]
    design = DesignSpace(space)

    return design


@pytest.fixture
def random_sampler(design5: DesignSpace):
    return RandomUniform(design=design5, seed=42)


@pytest.fixture
def latinhypercube_sampler(design5: DesignSpace):
    return LatinHypercube(design=design5, seed=42)


@pytest.fixture
def sobolsequence_sampler(design5: DesignSpace):
    return SobolSequence(design=design5, seed=42)
