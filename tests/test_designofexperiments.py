import pytest
from f3dasm.src.designofexperiments import DoE
from f3dasm.src.space import CategoricalSpace, ContinuousSpace, DiscreteSpace


def test_empty_space_doe():
    doe = DoE()
    empty_list = []
    assert doe.space == empty_list


def test_correct_doe():
    x1 = ContinuousSpace(name="x1", lower_bound=2.4, upper_bound=10.3)
    x2 = DiscreteSpace(name="x2", lower_bound=5, upper_bound=80)
    x3 = ContinuousSpace(name="x3", lower_bound=10.0, upper_bound=380.3)
    x4 = CategoricalSpace(name="x4", categories=["test1", "test2", "test3"])
    designspace = [x1, x2, x3, x4]

    doe = DoE(space=designspace)


def test_get_continuous_parameters():
    x1 = ContinuousSpace(name="x1", lower_bound=2.4, upper_bound=10.3)
    x2 = DiscreteSpace(name="x2", lower_bound=5, upper_bound=80)
    x3 = ContinuousSpace(name="x3", lower_bound=10.0, upper_bound=380.3)
    x4 = CategoricalSpace(name="x4", categories=["test1", "test2", "test3"])
    x5 = DiscreteSpace(name="x5", lower_bound=2, upper_bound=3)
    designspace = [x1, x2, x3, x4, x5]

    doe = DoE(space=designspace)
    assert doe.getContinuousParameters() == [x1, x3]


def test_get_discrete_parameters():
    x1 = ContinuousSpace(name="x1", lower_bound=2.4, upper_bound=10.3)
    x2 = DiscreteSpace(name="x2", lower_bound=5, upper_bound=80)
    x3 = ContinuousSpace(name="x3", lower_bound=10.0, upper_bound=380.3)
    x4 = CategoricalSpace(name="x4", categories=["test1", "test2", "test3"])
    x5 = DiscreteSpace(name="x5", lower_bound=2, upper_bound=3)
    designspace = [x1, x2, x3, x4, x5]

    doe = DoE(space=designspace)
    assert doe.getDiscreteParameters() == [x2, x5]


def test_get_categorical_parameters():
    x1 = ContinuousSpace(name="x1", lower_bound=2.4, upper_bound=10.3)
    x2 = DiscreteSpace(name="x2", lower_bound=5, upper_bound=80)
    x3 = ContinuousSpace(name="x3", lower_bound=10.0, upper_bound=380.3)
    x4 = CategoricalSpace(name="x4", categories=["test1", "test2", "test3"])
    x5 = DiscreteSpace(name="x5", lower_bound=2, upper_bound=3)
    designspace = [x1, x2, x3, x4, x5]

    doe = DoE(space=designspace)
    assert doe.getCategoricalParameters() == [x4]


def test_get_continuous_names():
    x1 = ContinuousSpace(name="x1", lower_bound=2.4, upper_bound=10.3)
    x2 = DiscreteSpace(name="x2", lower_bound=5, upper_bound=80)
    x3 = ContinuousSpace(name="x3", lower_bound=10.0, upper_bound=380.3)
    x4 = CategoricalSpace(name="x4", categories=["test1", "test2", "test3"])
    x5 = DiscreteSpace(name="x5", lower_bound=2, upper_bound=3)
    designspace = [x1, x2, x3, x4, x5]

    doe = DoE(space=designspace)
    assert doe.getContinuousNames() == ["x1", "x3"]


def test_get_discrete_names():
    x1 = ContinuousSpace(name="x1", lower_bound=2.4, upper_bound=10.3)
    x2 = DiscreteSpace(name="x2", lower_bound=5, upper_bound=80)
    x3 = ContinuousSpace(name="x3", lower_bound=10.0, upper_bound=380.3)
    x4 = CategoricalSpace(name="x4", categories=["test1", "test2", "test3"])
    x5 = DiscreteSpace(name="x5", lower_bound=2, upper_bound=3)
    designspace = [x1, x2, x3, x4, x5]

    doe = DoE(space=designspace)
    assert doe.getDiscreteNames() == ["x2", "x5"]


def test_get_categorical_names():
    x1 = ContinuousSpace(name="x1", lower_bound=2.4, upper_bound=10.3)
    x2 = DiscreteSpace(name="x2", lower_bound=5, upper_bound=80)
    x3 = ContinuousSpace(name="x3", lower_bound=10.0, upper_bound=380.3)
    x4 = CategoricalSpace(name="x4", categories=["test1", "test2", "test3"])
    x5 = DiscreteSpace(name="x5", lower_bound=2, upper_bound=3)
    x6 = CategoricalSpace(name="x6", categories=["material1", "material2", "material3"])
    designspace = [x1, x2, x3, x4, x5, x6]

    doe = DoE(space=designspace)
    assert doe.getCategoricalNames() == ["x4", "x6"]


def test_addspace():
    x1 = ContinuousSpace(name="x1", lower_bound=2.4, upper_bound=10.3)
    x2 = DiscreteSpace(name="x2", lower_bound=5, upper_bound=80)
    x3 = ContinuousSpace(name="x3", lower_bound=10.0, upper_bound=380.3)
    x4 = CategoricalSpace(name="x4", categories=["test1", "test2", "test3"])
    x5 = DiscreteSpace(name="x5", lower_bound=2, upper_bound=3)
    designspace = [x1, x2, x3]

    doe = DoE(space=designspace)
    doe.addSpace(x4)
    doe.addSpace(x5)

    assert doe.space == [x1, x2, x3, x4, x5]


def test_getNumberOfParameters():
    x1 = ContinuousSpace(name="x1", lower_bound=2.4, upper_bound=10.3)
    x2 = DiscreteSpace(name="x2", lower_bound=5, upper_bound=80)
    x3 = ContinuousSpace(name="x3", lower_bound=10.0, upper_bound=380.3)
    x4 = CategoricalSpace(name="x4", categories=["test1", "test2", "test3"])
    x5 = DiscreteSpace(name="x5", lower_bound=2, upper_bound=3)
    designspace = [x1, x2, x3, x4, x5]

    doe = DoE(space=designspace)

    assert doe.getNumberOfParameters() == 5


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
