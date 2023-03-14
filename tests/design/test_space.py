import numpy as np
import pytest

from f3dasm.design import (_create_parameter_from_dict,
                           create_parameter_from_json)
from f3dasm.design.parameter import (CategoricalParameter, ContinuousParameter,
                                     DiscreteParameter, Parameter)

pytestmark = pytest.mark.smoke


# Continuous space tests


def test_check_default_lower_bound():
    continuous = ContinuousParameter(name="test")
    assert continuous.lower_bound == -np.inf


def test_check_default_upper_bound():
    continuous = ContinuousParameter(name="test")
    assert continuous.upper_bound == np.inf


def test_correct_continuous_space():
    lower_bound = 3.3
    upper_bound = 3.8
    continuous = ContinuousParameter(name="test", lower_bound=lower_bound, upper_bound=upper_bound)


def test_higher_upper_bound_than_lower_bound_continuous_space():
    lower_bound = 2.0
    upper_bound = 1.5
    with pytest.raises(ValueError):
        continuous = ContinuousParameter(name="test", lower_bound=lower_bound, upper_bound=upper_bound)


def test_same_lower_and_upper_bound_continuous_space():
    lower_bound = 1.5
    upper_bound = 1.5
    with pytest.raises(ValueError):
        continuous = ContinuousParameter(name="test", lower_bound=lower_bound, upper_bound=upper_bound)


def test_invalid_types_arg1_int_continuous_space():
    lower_bound = 1  # int
    upper_bound = 1.5
    with pytest.raises(TypeError):
        continuous = ContinuousParameter(name="test", lower_bound=lower_bound, upper_bound=upper_bound)


def test_invalid_types_arg2_int_continuous_space():
    lower_bound = 1.0
    upper_bound = 2  # int
    with pytest.raises(TypeError):
        continuous = ContinuousParameter(name="test", lower_bound=lower_bound, upper_bound=upper_bound)


def test_invalid_types_string_continuous_space():
    lower_bound = "1"  # string
    upper_bound = 1.5
    with pytest.raises(TypeError):
        continuous = ContinuousParameter(name="test", lower_bound=lower_bound, upper_bound=upper_bound)


# Discrete space tests


def test_correct_discrete_space():
    lower_bound = 1
    upper_bound = 100
    continuous = DiscreteParameter(name="test", lower_bound=lower_bound, upper_bound=upper_bound)


def test_higher_upper_bound_than_lower_bound_discrete_space():
    lower_bound = 2
    upper_bound = 1
    with pytest.raises(ValueError):
        discrete = DiscreteParameter(name="test", lower_bound=lower_bound, upper_bound=upper_bound)


def test_same_lower_and_upper_bound_discrete_space():
    lower_bound = 1
    upper_bound = 1
    with pytest.raises(ValueError):
        discrete = DiscreteParameter(name="test", lower_bound=lower_bound, upper_bound=upper_bound)


def test_invalid_types_arg1_float_discrete_space():
    lower_bound = 1  # float
    upper_bound = 1.5
    with pytest.raises(TypeError):
        discrete = DiscreteParameter(name="test", lower_bound=lower_bound, upper_bound=upper_bound)


def test_invalid_types_arg2_float_discrete_space():
    lower_bound = 1
    upper_bound = 2.0  # float
    with pytest.raises(TypeError):
        discrete = DiscreteParameter(name="test", lower_bound=lower_bound, upper_bound=upper_bound)


def test_invalid_types_string_discrete_space():
    lower_bound = "1"  # string
    upper_bound = 1
    with pytest.raises(TypeError):
        discrete = DiscreteParameter(name="test", lower_bound=lower_bound, upper_bound=upper_bound)


# Categorical space tests


def test_correct_categorical_space():
    categories = ["test1", "test2", "test3", "test4"]
    continuous = CategoricalParameter(name="test", categories=categories)


def test_invalid_types_in_categories_categorical_space():
    categories = ["test1", "test2", 3, "test4"]
    with pytest.raises(TypeError):
        categorical = CategoricalParameter(name="test", categories=categories)


def test_invalid_types_categories_categorical_space():
    categories = ("test1", "test2", "test3")
    with pytest.raises(TypeError):
        categorical = CategoricalParameter(name="test", categories=categories)


def test_duplicates_categories_categorical_space():
    categories = ["test1", "test2", "test1"]
    with pytest.raises(ValueError):
        categorical = CategoricalParameter(name="test", categories=categories)


@pytest.mark.parametrize("parameter_name", ['categorical_parameter', 'discrete_parameter', 'continuous_parameter'])
def test_check_reproducibility(parameter_name: str, request):
    parameter: Parameter = request.getfixturevalue(parameter_name)
    json_string = parameter.to_json()
    parameter_new = create_parameter_from_json(json_string)
    assert parameter == parameter_new


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
