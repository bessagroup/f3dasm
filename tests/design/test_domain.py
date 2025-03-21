import json
from pathlib import Path

import numpy as np
import pytest

from f3dasm._src.design.domain import Domain
from f3dasm._src.design.parameter import (CategoricalParameter,
                                          ConstantParameter,
                                          ContinuousParameter,
                                          DiscreteParameter)

pytestmark = pytest.mark.smoke


# Continuous space tests


def test_check_default_lower_bound():
    continuous = ContinuousParameter()
    assert continuous.lower_bound == -np.inf


def test_check_default_upper_bound():
    continuous = ContinuousParameter()
    assert continuous.upper_bound == np.inf


def test_correct_continuous_space():
    lower_bound = 3.3
    upper_bound = 3.8
    _ = ContinuousParameter(lower_bound=lower_bound, upper_bound=upper_bound)


def test_higher_upper_bound_than_lower_bound_continuous_space():
    lower_bound = 2.0
    upper_bound = 1.5
    with pytest.raises(ValueError):
        _ = ContinuousParameter(
            lower_bound=lower_bound, upper_bound=upper_bound)


def test_same_lower_and_upper_bound_continuous_space():
    lower_bound = 1.5
    upper_bound = 1.5
    with pytest.raises(ValueError):
        _ = ContinuousParameter(
            lower_bound=lower_bound, upper_bound=upper_bound)


# Discrete space tests


def test_correct_discrete_space():
    lower_bound = 1
    upper_bound = 100
    _ = DiscreteParameter(lower_bound=lower_bound, upper_bound=upper_bound)


def test_higher_upper_bound_than_lower_bound_discrete_space():
    lower_bound = 2
    upper_bound = 1
    with pytest.raises(ValueError):
        _ = DiscreteParameter(lower_bound=lower_bound,
                              upper_bound=upper_bound)


def test_same_lower_and_upper_bound_discrete_space():
    lower_bound = 1
    upper_bound = 1
    with pytest.raises(ValueError):
        _ = DiscreteParameter(lower_bound=lower_bound,
                              upper_bound=upper_bound)


def test_integer_types_arg1_float_discrete_space():
    lower_bound = 1  # float
    upper_bound = 2.5
    parameter = DiscreteParameter(lower_bound=lower_bound,
                                  upper_bound=upper_bound)

    assert isinstance(parameter.lower_bound, int)


def test_float_types_arg2_float_discrete_space():
    lower_bound = 1
    upper_bound = 2.0  # float
    parameter = DiscreteParameter(lower_bound=lower_bound,
                                  upper_bound=upper_bound)
    assert isinstance(parameter.upper_bound, int)


# Categorical space tests


def test_correct_categorical_space():
    categories = ["test1", "test2", "test3", "test4"]
    _ = CategoricalParameter(categories=categories)


def test_invalid_types_in_categories_categorical_space():
    categories = ["test1", "test2", [3, 4], "test4"]
    with pytest.raises(TypeError):
        _ = CategoricalParameter(categories=categories)


def test_duplicates_categories_categorical_space():
    categories = ["test1", "test2", "test1"]
    with pytest.raises(ValueError):
        _ = CategoricalParameter(categories=categories)


@pytest.mark.parametrize("args", [((0., 5.), (-1., 3.), (-1., 5.),),
                                  ((0., 5.), (1., 3.), (0., 5.),),
                                  ((-1., 3.), (0., 5.), (-1., 5.),),
                                  ((0., 5.), (0., 5.), (0., 5.),)])
def test_add_continuous(args):
    a, b, expected = args
    param_a = ContinuousParameter(*a)
    param_b = ContinuousParameter(*b)

    assert param_a + param_b == ContinuousParameter(*expected)


@pytest.mark.parametrize("args", [((0., 5.), (6., 10.)),])
def test_faulty_continuous_ranges(args):
    a, b = args
    param_a = ContinuousParameter(*a)
    param_b = ContinuousParameter(*b)
    with pytest.raises(ValueError):
        param_a + param_b


def test_faulty_continous_log():
    a = ContinuousParameter(1., 5., log=True)
    b = ContinuousParameter(0., 5., log=False)
    with pytest.raises(ValueError):
        a + b


@pytest.mark.parametrize("args", [(('test1', 'test2'), ('test3',), ('test1', 'test2', 'test3'),),
                                  (('test1', 'test3'), ('test3',),
                                   ('test1', 'test3'),)])
def test_add_categorical(args):
    a, b, expected = args
    param_a = CategoricalParameter(list(a))
    param_b = CategoricalParameter(list(b))

    assert param_a + param_b == CategoricalParameter(list(expected))


@pytest.mark.parametrize(
    "args",
    [(CategoricalParameter(['test1', 'test2']), ConstantParameter('test3'), CategoricalParameter(['test1', 'test2', 'test3']),),
     (CategoricalParameter(['test1', 'test2']), DiscreteParameter(
         1, 3), CategoricalParameter(['test1', 'test2', 1, 2]),),
     (CategoricalParameter(['test1', 'test2']), ConstantParameter(
         'test1'), CategoricalParameter(['test1', 'test2']),),
     (CategoricalParameter(['test1', 'test2']), CategoricalParameter([
         'test1']), CategoricalParameter(['test1', 'test2']),),
        (ConstantParameter('test3'), CategoricalParameter(
            ['test1', 'test2']), CategoricalParameter(['test1', 'test2', 'test3']))


     ])
def test_add_combination(args):
    a, b, expected = args
    assert a + b == expected


def test_to_discrete():
    a = ContinuousParameter(0., 5.)
    c = DiscreteParameter(0, 5, 0.2)
    b = a.to_discrete(0.2)
    assert isinstance(b, DiscreteParameter)
    assert b.lower_bound == 0
    assert b.upper_bound == 5
    assert b.step == 0.2
    assert b == c


def test_to_discrete_negative_stepsize():
    a = ContinuousParameter(0., 5.)
    with pytest.raises(ValueError):
        a.to_discrete(-0.2)


def test_default_stepsize_to_discrete():
    default_stepsize = 1
    a = ContinuousParameter(0., 5.)
    c = DiscreteParameter(0, 5, default_stepsize)
    b = a.to_discrete()
    assert isinstance(b, DiscreteParameter)
    assert b.lower_bound == 0
    assert b.upper_bound == 5
    assert b.step == default_stepsize
    assert b == c


def test_domain_store(tmp_path):
    domain = Domain()
    domain.add_float('param1', 0.0, 1.0)
    domain.add_int('param2', 0, 10)
    domain.add_category('param3', ['a', 'b', 'c'])
    domain.add_constant('param4', 42)

    json_file = tmp_path / "domain.json"
    domain.store(json_file)

    with open(json_file, 'r') as f:
        data = json.load(f)

    assert 'input_space' in data
    assert 'output_space' in data
    assert 'param1' in data['input_space']
    assert 'param2' in data['input_space']
    assert 'param3' in data['input_space']
    assert 'param4' in data['input_space']


def test_domain_from_json(tmp_path):
    domain = Domain()
    domain.add_float('param1', 0.0, 1.0)
    domain.add_int('param2', 0, 10)
    domain.add_category('param3', ['a', 'b', 'c'])
    domain.add_constant('param4', 42)

    json_file = tmp_path / "domain.json"
    domain.store(json_file)

    loaded_domain = Domain.from_file(json_file)

    assert loaded_domain.input_space['param1'] == domain.input_space['param1']
    assert loaded_domain.input_space['param2'] == domain.input_space['param2']
    assert loaded_domain.input_space['param3'] == domain.input_space['param3']
    assert loaded_domain.input_space['param4'] == domain.input_space['param4']


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
