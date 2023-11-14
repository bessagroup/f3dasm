import numpy as np
import pandas as pd
import pytest

from f3dasm.design import (Domain, _CategoricalParameter, _ContinuousParameter,
                           _DiscreteParameter, make_nd_continuous_domain)

pytestmark = pytest.mark.smoke


def test_empty_space_doe():
    doe = Domain()
    empty_dict = {}
    assert doe.space == empty_dict


def test_correct_doe(doe):
    pass


def test_get_continuous_parameters(doe: Domain):
    design = {'x1': _ContinuousParameter(lower_bound=2.4, upper_bound=10.3),
              'x3': _ContinuousParameter(lower_bound=10.0, upper_bound=380.3)}
    assert doe.get_continuous_parameters() == design


def test_get_discrete_parameters(doe: Domain):
    design = {'x2': _DiscreteParameter(lower_bound=5, upper_bound=80),
              'x5': _DiscreteParameter(lower_bound=2, upper_bound=3)}
    assert doe.get_discrete_parameters() == design


def test_get_categorical_parameters(doe: Domain):
    assert doe.get_categorical_parameters() == {'x4': _CategoricalParameter(
        categories=["test1", "test2", "test3"])}


def test_get_continuous_names(doe: Domain):
    assert doe.get_continuous_names() == ["x1", "x3"]


def test_get_discrete_names(doe: Domain):
    assert doe.get_discrete_names() == ["x2", "x5"]


def test_get_categorical_names(doe: Domain):
    assert doe.get_categorical_names() == ["x4"]


def test_add_arbitrary_list_as_categorical_parameter():
    arbitrary_list_1 = [3.1416, "pi", 42]
    arbitrary_list_2 = np.linspace(start=142, stop=214, num=10)
    designspace = {'x1': _CategoricalParameter(categories=arbitrary_list_1),
                   'x2': _CategoricalParameter(categories=arbitrary_list_2)
                   }

    design = Domain(space=designspace)

    assert design.space == designspace


def test_add_input_space():
    designspace = {
        'x1': _ContinuousParameter(lower_bound=2.4, upper_bound=10.3),
        'x2': _DiscreteParameter(lower_bound=5, upper_bound=80),
        'x3': _ContinuousParameter(lower_bound=10.0, upper_bound=380.3),
    }

    design = Domain(space=designspace)
    design.add('x4', type='category',
               categories=["test1", "test2", "test3"])
    design.add('x5', type='int', low=2, high=3)

    assert design.space == {
        'x1': _ContinuousParameter(lower_bound=2.4, upper_bound=10.3),
        'x2': _DiscreteParameter(lower_bound=5, upper_bound=80),
        'x3': _ContinuousParameter(lower_bound=10.0, upper_bound=380.3),
        'x4': _CategoricalParameter(categories=["test1", "test2", "test3"]),
        'x5': _DiscreteParameter(lower_bound=2, upper_bound=3)
    }


def test_add_space():
    designspace = {
        'x1': _ContinuousParameter(lower_bound=2.4, upper_bound=10.3),
        'x2': _DiscreteParameter(lower_bound=5, upper_bound=80),
        'x3': _ContinuousParameter(lower_bound=10.0, upper_bound=380.3),
    }

    domain = Domain(space=designspace)
    domain.add('x4', type='category',
               categories=["test1", "test2", "test3"])
    domain.add('x5', type='int', low=2, high=3)

    assert domain.space == {
        'x1': _ContinuousParameter(lower_bound=2.4, upper_bound=10.3),
        'x2': _DiscreteParameter(lower_bound=5, upper_bound=80),
        'x3': _ContinuousParameter(lower_bound=10.0, upper_bound=380.3),
        'x4': _CategoricalParameter(categories=["test1", "test2", "test3"]),
        'x5': _DiscreteParameter(lower_bound=2, upper_bound=3),
    }


def test_getNumberOfInputParameters(doe: Domain):
    assert len(doe) == 5


def test_all_input_continuous_False(doe: Domain):
    assert doe._all_input_continuous() is False


def test_all_input_continuous_True():
    designspace = {
        'x1': _ContinuousParameter(lower_bound=2.4, upper_bound=10.3),
        'x3': _ContinuousParameter(lower_bound=10.0, upper_bound=380.3),
    }

    doe = Domain(space=designspace)

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


def test_get_input_names(domain: Domain):
    # Ensure that get_input_names returns the correct input parameter names
    assert domain.names == ['x1', 'x2', 'x3']


def test_get_number_of_input_parameters(domain: Domain):
    # Ensure that get_number_of_input_parameters returns the correct number of input parameters
    assert len(domain) == 3


def test_domain_from_dataframe(sample_dataframe: pd.DataFrame):
    domain = Domain.from_dataframe(
        df_input=sample_dataframe, df_output=pd.DataFrame())
    ground_truth = Domain(space={'feature1': _ContinuousParameter(lower_bound=1.0, upper_bound=3.0),
                                 'feature2': _DiscreteParameter(lower_bound=4, upper_bound=6),
                                 'feature3': _CategoricalParameter(['A', 'B', 'C'])})
    assert (domain.space == ground_truth.space)


@pytest.mark.parametrize("bounds", [((0., 1.), (0., 1.), (0., 1.)), ([0., 1.], [0., 1.], [0., 1.]),
                                    np.array([[0., 1.], [0., 1.], [0., 1.]]), np.tile([0., 1.], (3, 1))])
def test_make_nd_continuous_domain(bounds):
    domain = make_nd_continuous_domain(bounds=bounds, dimensionality=3)
    ground_truth = Domain(space={'x0': _ContinuousParameter(lower_bound=0.0, upper_bound=1.0),
                                 'x1': _ContinuousParameter(lower_bound=0.0, upper_bound=1.0),
                                 'x2': _ContinuousParameter(lower_bound=0.0, upper_bound=1.0)})
    assert (domain.space == ground_truth.space)


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
