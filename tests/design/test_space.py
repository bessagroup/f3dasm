import numpy as np
import pytest

from f3dasm.design import (_CategoricalParameter, _ContinuousParameter,
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


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
