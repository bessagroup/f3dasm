import numpy as np
import pytest

from f3dasm._src.design.parameter import (_CategoricalParameter,
                                          _ConstantParameter,
                                          _ContinuousParameter,
                                          _DiscreteParameter)

pytestmark = pytest.mark.smoke


# Continuous space tests


def test_check_default_lower_bound():
    continuous = _ContinuousParameter()
    assert continuous.lower_bound == -np.inf


def test_check_default_upper_bound():
    continuous = _ContinuousParameter()
    assert continuous.upper_bound == np.inf


def test_correct_continuous_space():
    lower_bound = 3.3
    upper_bound = 3.8
    _ = _ContinuousParameter(lower_bound=lower_bound, upper_bound=upper_bound)


def test_higher_upper_bound_than_lower_bound_continuous_space():
    lower_bound = 2.0
    upper_bound = 1.5
    with pytest.raises(ValueError):
        _ = _ContinuousParameter(
            lower_bound=lower_bound, upper_bound=upper_bound)


def test_same_lower_and_upper_bound_continuous_space():
    lower_bound = 1.5
    upper_bound = 1.5
    with pytest.raises(ValueError):
        _ = _ContinuousParameter(
            lower_bound=lower_bound, upper_bound=upper_bound)


def test_invalid_types_string_continuous_space():
    lower_bound = "1"  # string
    upper_bound = 1.5
    with pytest.raises(TypeError):
        _ = _ContinuousParameter(
            lower_bound=lower_bound, upper_bound=upper_bound)


# Discrete space tests


def test_correct_discrete_space():
    lower_bound = 1
    upper_bound = 100
    _ = _DiscreteParameter(lower_bound=lower_bound, upper_bound=upper_bound)


def test_higher_upper_bound_than_lower_bound_discrete_space():
    lower_bound = 2
    upper_bound = 1
    with pytest.raises(ValueError):
        _ = _DiscreteParameter(lower_bound=lower_bound,
                               upper_bound=upper_bound)


def test_same_lower_and_upper_bound_discrete_space():
    lower_bound = 1
    upper_bound = 1
    with pytest.raises(ValueError):
        _ = _DiscreteParameter(lower_bound=lower_bound,
                               upper_bound=upper_bound)


def test_invalid_types_arg1_float_discrete_space():
    lower_bound = 1  # float
    upper_bound = 1.5
    with pytest.raises(TypeError):
        _ = _DiscreteParameter(lower_bound=lower_bound,
                               upper_bound=upper_bound)


def test_invalid_types_arg2_float_discrete_space():
    lower_bound = 1
    upper_bound = 2.0  # float
    with pytest.raises(TypeError):
        _ = _DiscreteParameter(lower_bound=lower_bound,
                               upper_bound=upper_bound)


def test_invalid_types_string_discrete_space():
    lower_bound = "1"  # string
    upper_bound = 1
    with pytest.raises(TypeError):
        _ = _DiscreteParameter(lower_bound=lower_bound,
                               upper_bound=upper_bound)


# Categorical space tests


def test_correct_categorical_space():
    categories = ["test1", "test2", "test3", "test4"]
    _ = _CategoricalParameter(categories=categories)


def test_invalid_types_in_categories_categorical_space():
    categories = ["test1", "test2", [3, 4], "test4"]
    with pytest.raises(TypeError):
        _ = _CategoricalParameter(categories=categories)


def test_duplicates_categories_categorical_space():
    categories = ["test1", "test2", "test1"]
    with pytest.raises(ValueError):
        _ = _CategoricalParameter(categories=categories)


@pytest.mark.parametrize("args", [((0., 5.), (-1., 3.), (-1., 5.),),
                                  ((0., 5.), (1., 3.), (0., 5.),),
                                  ((-1., 3.), (0., 5.), (-1., 5.),),
                                  ((0., 5.), (0., 5.), (0., 5.),)])
def test_add_continuous(args):
    a, b, expected = args
    param_a = _ContinuousParameter(*a)
    param_b = _ContinuousParameter(*b)

    assert param_a + param_b == _ContinuousParameter(*expected)


@pytest.mark.parametrize("args", [((0., 5.), (6., 10.)),])
def test_faulty_continuous_ranges(args):
    a, b = args
    param_a = _ContinuousParameter(*a)
    param_b = _ContinuousParameter(*b)
    with pytest.raises(ValueError):
        param_a + param_b


def test_faulty_continous_log():
    a = _ContinuousParameter(1., 5., log=True)
    b = _ContinuousParameter(0., 5., log=False)
    with pytest.raises(ValueError):
        a + b


@pytest.mark.parametrize("args", [(('test1', 'test2'), ('test3',), ('test1', 'test2', 'test3'),),
                                  (('test1', 'test3'), ('test3',),
                                   ('test1', 'test3'),)])
def test_add_categorical(args):
    a, b, expected = args
    param_a = _CategoricalParameter(list(a))
    param_b = _CategoricalParameter(list(b))

    assert param_a + param_b == _CategoricalParameter(list(expected))


@pytest.mark.parametrize(
    "args",
    [(_CategoricalParameter(['test1', 'test2']), _ConstantParameter('test3'), _CategoricalParameter(['test1', 'test2', 'test3']),),
     (_CategoricalParameter(['test1', 'test2']), _DiscreteParameter(
         1, 3), _CategoricalParameter(['test1', 'test2', 1, 2]),),
     (_CategoricalParameter(['test1', 'test2']), _ConstantParameter(
         'test1'), _CategoricalParameter(['test1', 'test2']),),
     (_CategoricalParameter(['test1', 'test2']), _CategoricalParameter([
         'test1']), _CategoricalParameter(['test1', 'test2']),),
        (_ConstantParameter('test3'), _CategoricalParameter(
            ['test1', 'test2']), _CategoricalParameter(['test1', 'test2', 'test3']))


     ])
def test_add_combination(args):
    a, b, expected = args
    assert a + b == expected


def test_to_discrete():
    a = _ContinuousParameter(0., 5.)
    c = _DiscreteParameter(0, 5, 0.2)
    b = a.to_discrete(0.2)
    assert isinstance(b, _DiscreteParameter)
    assert b.lower_bound == 0
    assert b.upper_bound == 5
    assert b.step == 0.2
    assert b == c


def test_to_discrete_negative_stepsize():
    a = _ContinuousParameter(0., 5.)
    with pytest.raises(ValueError):
        a.to_discrete(-0.2)


def test_default_stepsize_to_discrete():
    default_stepsize = 1
    a = _ContinuousParameter(0., 5.)
    c = _DiscreteParameter(0, 5, default_stepsize)
    b = a.to_discrete()
    assert isinstance(b, _DiscreteParameter)
    assert b.lower_bound == 0
    assert b.upper_bound == 5
    assert b.step == default_stepsize
    assert b == c


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
