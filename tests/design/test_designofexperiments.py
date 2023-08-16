import json

import numpy as np
import pandas as pd
import pytest

from f3dasm.design.domain import Domain
from f3dasm.design.parameter import (CategoricalParameter, ContinuousParameter,
                                     DiscreteParameter)

pytestmark = pytest.mark.smoke


def test_empty_space_doe():
    doe = Domain()
    empty_dict = {}
    assert doe.input_space == empty_dict


def test_correct_doe(doe):
    pass


def test_get_continuous_parameters(doe: Domain):
    design = {'x1': ContinuousParameter(lower_bound=2.4, upper_bound=10.3),
              'x3': ContinuousParameter(lower_bound=10.0, upper_bound=380.3)}
    assert doe.get_continuous_input_parameters() == design


def test_get_discrete_parameters(doe: Domain):
    design = {'x2': DiscreteParameter(lower_bound=5, upper_bound=80),
              'x5': DiscreteParameter(lower_bound=2, upper_bound=3)}
    assert doe.get_discrete_input_parameters() == design


def test_get_categorical_parameters(doe: Domain):
    assert doe.get_categorical_input_parameters() == {'x4': CategoricalParameter(
        categories=["test1", "test2", "test3"])}


def test_get_continuous_names(doe: Domain):
    assert doe.get_continuous_input_names() == ["x1", "x3"]


def test_get_discrete_names(doe: Domain):
    assert doe.get_discrete_input_names() == ["x2", "x5"]


def test_get_categorical_names(doe: Domain):
    assert doe.get_categorical_input_names() == ["x4"]


def test_add_arbitrary_list_as_categorical_parameter():
    arbitrary_list_1 = [3.1416, "pi", 42]
    arbitrary_list_2 = np.linspace(start=142, stop=214, num=10)
    designspace = {'x1': CategoricalParameter(categories=arbitrary_list_1),
                   'x2': CategoricalParameter(categories=arbitrary_list_2)
                   }

    design = Domain(input_space=designspace)

    assert design.input_space == designspace


def test_add_input_space():
    designspace = {
        'x1': ContinuousParameter(lower_bound=2.4, upper_bound=10.3),
        'x2': DiscreteParameter(lower_bound=5, upper_bound=80),
        'x3': ContinuousParameter(lower_bound=10.0, upper_bound=380.3),
    }

    design = Domain(input_space=designspace)
    design.add_input_space('x4', CategoricalParameter(categories=["test1", "test2", "test3"]))
    design.add_input_space('x5', DiscreteParameter(lower_bound=2, upper_bound=3))

    assert design.input_space == {
        'x1': ContinuousParameter(lower_bound=2.4, upper_bound=10.3),
        'x2': DiscreteParameter(lower_bound=5, upper_bound=80),
        'x3': ContinuousParameter(lower_bound=10.0, upper_bound=380.3),
        'x4': CategoricalParameter(categories=["test1", "test2", "test3"]),
        'x5': DiscreteParameter(lower_bound=2, upper_bound=3)
    }


def test_add_space():
    designspace = {
        'x1': ContinuousParameter(lower_bound=2.4, upper_bound=10.3),
        'x2': DiscreteParameter(lower_bound=5, upper_bound=80),
        'x3': ContinuousParameter(lower_bound=10.0, upper_bound=380.3),
    }

    domain = Domain(input_space=designspace)
    domain.add_input_space('x4', CategoricalParameter(categories=["test1", "test2", "test3"]))
    domain.add_input_space('x5', DiscreteParameter(lower_bound=2, upper_bound=3))

    assert domain.input_space == {
        'x1': ContinuousParameter(lower_bound=2.4, upper_bound=10.3),
        'x2': DiscreteParameter(lower_bound=5, upper_bound=80),
        'x3': ContinuousParameter(lower_bound=10.0, upper_bound=380.3),
        'x4': CategoricalParameter(categories=["test1", "test2", "test3"]),
        'x5': DiscreteParameter(lower_bound=2, upper_bound=3),
    }


def test_getNumberOfInputParameters(doe: Domain):
    assert doe.get_number_of_input_parameters() == 5


def test_get_input_space(doe: Domain):
    assert doe.input_space == doe.get_input_space()


def test_all_input_continuous_False(doe: Domain):
    assert doe._all_input_continuous() is False


def test_all_input_continuous_True():
    designspace = {
        'x1': ContinuousParameter(lower_bound=2.4, upper_bound=10.3),
        'x3': ContinuousParameter(lower_bound=10.0, upper_bound=380.3),
    }

    doe = Domain(input_space=designspace)

    assert doe._all_input_continuous() is True


def test_cast_types_dataframe_input(doe: Domain):
    ground_truth = {
        "x1": "float",
        "x2": "int",
        "x3": "float",
        "x4": "category",
        "x5": "int",
    }

    assert doe._cast_types_dataframe() == ground_truth


def test_get_input_space_2(domain: Domain):
    # Ensure that get_input_space returns the input space
    assert domain.get_input_space() == domain.input_space


def test_get_input_names(domain: Domain):
    # Ensure that get_input_names returns the correct input parameter names
    assert domain.get_input_names() == ['x1', 'x2', 'x3']


def test_get_number_of_input_parameters(domain: Domain):
    # Ensure that get_number_of_input_parameters returns the correct number of input parameters
    assert domain.get_number_of_input_parameters() == 3


def test_domain_from_dataframe(sample_dataframe: pd.DataFrame):
    domain = Domain.from_dataframe(sample_dataframe)
    ground_truth = Domain(input_space={'feature1': ContinuousParameter(lower_bound=1.0, upper_bound=3.0),
                                       'feature2': DiscreteParameter(lower_bound=4, upper_bound=6),
                                       'feature3': CategoricalParameter(['A', 'B', 'C'])})
    assert (domain.input_space == ground_truth.input_space)


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
