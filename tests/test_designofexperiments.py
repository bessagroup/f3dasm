import pytest
from f3dasm.src.designofexperiments import DoE
from f3dasm.src.space import CategoricalSpace, ContinuousSpace, DiscreteSpace


@pytest.fixture
def doe():
    x1 = ContinuousSpace(name="x1", lower_bound=2.4, upper_bound=10.3)
    x2 = DiscreteSpace(name="x2", lower_bound=5, upper_bound=80)
    x3 = ContinuousSpace(name="x3", lower_bound=10.0, upper_bound=380.3)
    x4 = CategoricalSpace(name="x4", categories=["test1", "test2", "test3"])
    x5 = DiscreteSpace(name="x5", lower_bound=2, upper_bound=3)
    designspace = [x1, x2, x3, x4, x5]

    doe = DoE(input_space=designspace)
    return doe


def test_empty_space_doe():
    doe = DoE()
    empty_list = []
    assert doe.input_space == empty_list


def test_correct_doe(doe):
    pass


def test_get_continuous_parameters(doe):
    x1 = ContinuousSpace(name="x1", lower_bound=2.4, upper_bound=10.3)
    x3 = ContinuousSpace(name="x3", lower_bound=10.0, upper_bound=380.3)
    assert doe.getContinuousParameters() == [x1, x3]


def test_get_discrete_parameters(doe):
    x2 = DiscreteSpace(name="x2", lower_bound=5, upper_bound=80)
    x5 = DiscreteSpace(name="x5", lower_bound=2, upper_bound=3)
    assert doe.getDiscreteParameters() == [x2, x5]


def test_get_categorical_parameters(doe):
    x4 = CategoricalSpace(name="x4", categories=["test1", "test2", "test3"])
    assert doe.getCategoricalParameters() == [x4]


def test_get_continuous_names(doe):
    assert doe.getContinuousNames() == ["x1", "x3"]


def test_get_discrete_names(doe):
    assert doe.getDiscreteNames() == ["x2", "x5"]


def test_get_categorical_names(doe):
    assert doe.getCategoricalNames() == ["x4"]


def test_add_input_space():
    x1 = ContinuousSpace(name="x1", lower_bound=2.4, upper_bound=10.3)
    x2 = DiscreteSpace(name="x2", lower_bound=5, upper_bound=80)
    x3 = ContinuousSpace(name="x3", lower_bound=10.0, upper_bound=380.3)
    x4 = CategoricalSpace(name="x4", categories=["test1", "test2", "test3"])
    x5 = DiscreteSpace(name="x5", lower_bound=2, upper_bound=3)
    designspace = [x1, x2, x3]

    design = DoE(input_space=designspace)
    design.add_input_space(x4)
    design.add_input_space(x5)

    assert design.input_space == [x1, x2, x3, x4, x5]

def test_add_output_space():
    x1 = ContinuousSpace(name="x1", lower_bound=2.4, upper_bound=10.3)
    x2 = DiscreteSpace(name="x2", lower_bound=5, upper_bound=80)
    x3 = ContinuousSpace(name="x3", lower_bound=10.0, upper_bound=380.3)
    x4 = CategoricalSpace(name="x4", categories=["test1", "test2", "test3"])
    x5 = DiscreteSpace(name="x5", lower_bound=2, upper_bound=3)
    designspace = [x1, x2, x3]

    design = DoE(output_space=designspace)
    design.add_output_space(x4)
    design.add_output_space(x5)

    assert design.output_space == [x1, x2, x3, x4, x5]


def test_getNumberOfInputParameters(doe):
    assert doe.getNumberOfInputParameters() == 5


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
