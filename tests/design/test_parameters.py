import pytest

from f3dasm._src.design.parameter import (  # Replace with actual module name
    CategoricalParameter, ConstantParameter, ContinuousParameter,
    DiscreteParameter, Parameter)

pytestmark = pytest.mark.smoke


# Test Parameter Base Class
@pytest.mark.parametrize("to_disk, store_function, load_function, raises", [
    (False, None, None, False),
    (True, lambda x, y: None, lambda x: None, False),
    (False, lambda x, y: None, None, True),
    (False, None, lambda x: None, True),
])
def test_parameter_init(to_disk, store_function, load_function, raises):
    if raises:
        with pytest.raises(ValueError):
            Parameter(to_disk, store_function, load_function)
    else:
        param = Parameter(to_disk, store_function, load_function)
        assert param.to_disk == to_disk
        assert param.store_function == store_function
        assert param.load_function == load_function


def test_parameter_str():
    param = Parameter()
    assert str(param) == "Parameter(type=object, to_disk=False)"


def test_parameter_repr():
    param = Parameter()
    assert repr(param) == "Parameter(to_disk=False)"


def test_parameter_eq():
    param1 = Parameter()
    param2 = Parameter()
    assert param1 == param2


def test_parameter_add():
    param1 = Parameter()
    param2 = Parameter()
    result = param1 + param2
    assert result == param1

# Test ConstantParameter


@pytest.mark.parametrize("value, raises", [
    (5, False),
    ("test", False),
    ({"a": 1}, True),
])
def test_constant_parameter_init(value, raises):
    if raises:
        with pytest.raises(TypeError):
            ConstantParameter(value)
    else:
        param = ConstantParameter(value)
        assert param.value == value
        assert param.to_disk is False


def test_constant_parameter_to_categorical():
    param = ConstantParameter(42)
    cat_param = param.to_categorical()
    assert isinstance(cat_param, CategoricalParameter)
    assert cat_param.categories == [42]


def test_constant_parameter_str():
    param = ConstantParameter(42)
    assert str(param) == "ConstantParameter(value=42)"


def test_constant_parameter_repr():
    param = ConstantParameter(42)
    assert repr(param) == "ConstantParameter(value=42)"


def test_constant_parameter_eq():
    param1 = ConstantParameter(42)
    param2 = ConstantParameter(42)
    assert param1 == param2

# Test ContinuousParameter


@pytest.mark.parametrize("lower_bound, upper_bound, log, raises", [
    (0.0, 10.0, False, False),
    (0.0, 10.0, True, True),
    (0.0, 0.0, False, True),
    (10.0, 5.0, False, True),
    (-1.0, 5.0, True, True),
])
def test_continuous_parameter_init(lower_bound, upper_bound, log, raises):
    if raises:
        with pytest.raises(ValueError):
            ContinuousParameter(lower_bound, upper_bound, log)
    else:
        param = ContinuousParameter(lower_bound, upper_bound, log)
        assert param.lower_bound == lower_bound
        assert param.upper_bound == upper_bound
        assert param.log == log


def test_continuous_parameter_to_discrete():
    param = ContinuousParameter(0.0, 10.0)
    discrete_param = param.to_discrete(step=2)
    assert isinstance(discrete_param, DiscreteParameter)
    assert discrete_param.lower_bound == 0.0
    assert discrete_param.upper_bound == 10.0
    assert discrete_param.step == 2


def test_continuous_parameter_str():
    param = ContinuousParameter(0.0, 10.0, log=False)
    assert str(
        param) == "ContinuousParameter(lower_bound=0.0, upper_bound=10.0, log=False)"


def test_continuous_parameter_repr():
    param = ContinuousParameter(0.0, 10.0, log=False)
    assert repr(
        param) == "ContinuousParameter(lower_bound=0.0, upper_bound=10.0, log=False)"


def test_continuous_parameter_eq():
    param1 = ContinuousParameter(0.0, 10.0, log=False)
    param2 = ContinuousParameter(0.0, 10.0, log=False)
    assert param1 == param2

# Test DiscreteParameter


@pytest.mark.parametrize("lower_bound, upper_bound, step, raises", [
    (0, 10, 1, False),
    (10, 0, 1, True),
    (0, 10, -1, True),
])
def test_discrete_parameter_init(lower_bound, upper_bound, step, raises):
    if raises:
        with pytest.raises(ValueError):
            DiscreteParameter(lower_bound, upper_bound, step)
    else:
        param = DiscreteParameter(lower_bound, upper_bound, step)
        assert param.lower_bound == lower_bound
        assert param.upper_bound == upper_bound
        assert param.step == step


def test_discrete_parameter_str():
    param = DiscreteParameter(0, 10, 2)
    assert str(param) == "DiscreteParameter(lower_bound=0, upper_bound=10, step=2)"


def test_discrete_parameter_repr():
    param = DiscreteParameter(0, 10, 2)
    assert repr(
        param) == "DiscreteParameter(lower_bound=0, upper_bound=10, step=2)"


def test_discrete_parameter_eq():
    param1 = DiscreteParameter(0, 10, 2)
    param2 = DiscreteParameter(0, 10, 2)
    assert param1 == param2

# Test CategoricalParameter


@pytest.mark.parametrize("categories, raises", [
    (["a", "b", "c"], False),
    ([1, 2, 3], False),
    (["a", "a", "b"], True),
])
def test_categorical_parameter_init(categories, raises):
    if raises:
        with pytest.raises(ValueError):
            CategoricalParameter(categories)
    else:
        param = CategoricalParameter(categories)
        assert set(param.categories) == set(categories)


def test_categorical_parameter_str():
    param = CategoricalParameter(["a", "b", "c"])
    assert str(param) == "CategoricalParameter(categories=['a', 'b', 'c'])"


def test_categorical_parameter_repr():
    param = CategoricalParameter(["a", "b", "c"])
    assert repr(param) == "CategoricalParameter(categories=['a', 'b', 'c'])"


def test_categorical_parameter_eq():
    param1 = CategoricalParameter(["a", "b", "c"])
    param2 = CategoricalParameter(["c", "b", "a"])
    assert param1 == param2


def test_add_constant_to_constant():
    param1 = ConstantParameter(5)
    param2 = ConstantParameter(5)
    result = param1 + param2
    assert result == param1


def test_add_constant_to_categorical():
    const_param = ConstantParameter(5)
    cat_param = CategoricalParameter([1, 2, 3])
    result = const_param + cat_param
    assert isinstance(result, CategoricalParameter)
    assert set(result.categories) == {1, 2, 3, 5}


def test_add_categorical_to_categorical():
    cat_param1 = CategoricalParameter(["a", "b"])
    cat_param2 = CategoricalParameter(["b", "c"])
    result = cat_param1 + cat_param2
    assert isinstance(result, CategoricalParameter)
    assert set(result.categories) == {"a", "b", "c"}


def test_add_continuous_to_continuous():
    param1 = ContinuousParameter(0.0, 10.0)
    param2 = ContinuousParameter(5.0, 15.0)
    result = param1 + param2
    assert isinstance(result, ContinuousParameter)
    assert result.lower_bound == 0.0
    assert result.upper_bound == 15.0
