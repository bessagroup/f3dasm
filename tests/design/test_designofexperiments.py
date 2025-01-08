import numpy as np
import pytest

from f3dasm._src.design.domain import _domain_factory
from f3dasm._src.design.parameter import (CategoricalParameter,
                                          ConstantParameter,
                                          ContinuousParameter,
                                          DiscreteParameter)
from f3dasm.design import Domain, make_nd_continuous_domain

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
    assert doe.continuous.input_space == design


def test_get_discrete_parameters(doe: Domain):
    design = {'x2': DiscreteParameter(lower_bound=5, upper_bound=80),
              'x5': DiscreteParameter(lower_bound=2, upper_bound=3)}
    assert doe.discrete.input_space == design


def test_get_categorical_parameters(doe: Domain):
    assert doe.categorical.input_space == {'x4': CategoricalParameter(
        categories=["test1", "test2", "test3"])}


def test_get_continuous_names(doe: Domain):
    assert doe.continuous.input_names == ["x1", "x3"]


def test_get_discrete_names(doe: Domain):
    assert doe.discrete.input_names == ["x2", "x5"]


def test_get_categorical_names(doe: Domain):
    assert doe.categorical.input_names == ["x4"]


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
    design.add('x4', type='category',
               categories=["test1", "test2", "test3"])
    design.add('x5', type='int', low=2, high=3)

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
    domain.add('x4', type='category',
               categories=["test1", "test2", "test3"])
    domain.add('x5', type='int', low=2, high=3)

    assert domain.input_space == {
        'x1': ContinuousParameter(lower_bound=2.4, upper_bound=10.3),
        'x2': DiscreteParameter(lower_bound=5, upper_bound=80),
        'x3': ContinuousParameter(lower_bound=10.0, upper_bound=380.3),
        'x4': CategoricalParameter(categories=["test1", "test2", "test3"]),
        'x5': DiscreteParameter(lower_bound=2, upper_bound=3),
    }


def test_getNumberOfInputParameters(doe: Domain):
    assert len(doe) == 5


def test_cast_types_dataframe_input(doe: Domain):
    ground_truth = {
        "x1": "float",
        "x2": "int",
        "x3": "float",
        "x4": "object",
        "x5": "int",
    }

    assert doe._cast_types_dataframe() == ground_truth


def test_get_input_names(domain: Domain):
    # Ensure that get_input_names returns the correct input parameter names
    assert domain.input_names == ['x1', 'x2', 'x3']


def test_get_number_of_input_parameters(domain: Domain):
    # Ensure that get_number_of_input_parameters returns the correct number of input parameters
    assert len(domain) == 3


def test_add_parameter_with_same_name_error():
    domain = Domain()
    domain.add('x1', type='float', low=0.0, high=1.0)
    with pytest.raises(KeyError):
        domain.add('x1', type='int', low=0, high=1)


def test_add_int_with_same_low_and_high():
    domain = Domain()
    domain.add_int('x1', low=1, high=1)
    assert domain.input_space == {
        'x1': ConstantParameter(value=1)}


def test_add_float_with_same_low_and_high():
    domain = Domain()
    domain.add_float('x1', low=1.0, high=1.0)
    assert domain.input_space == {
        'x1': ConstantParameter(value=1.0)}


def test_add_category_with_add_method():
    domain = Domain()
    domain.add('x1', type='category', categories=['a', 'b', 'c'])
    assert domain.input_space == {
        'x1': CategoricalParameter(categories=['a', 'b', 'c'])}


def test_add_constant_with_add_method():
    domain = Domain()
    domain.add('x1', type='constant', value=1)
    assert domain.input_space == {
        'x1': ConstantParameter(value=1)}


def test_add_output_that_exists_without_exist_ok():
    domain = Domain()
    domain.add_output('y')
    with pytest.raises(KeyError):
        domain.add_output('y', exist_ok=False)


def test_add_output_that_exists_with_exist_ok():
    domain = Domain()
    domain.add_output('y')
    domain.add_output('y', exist_ok=True)
    assert domain.output_names == ['y']


def test_domain_factory_with_invalid_input():
    with pytest.raises(TypeError):
        _domain_factory(domain=0)


def test_eq_different_types():
    domain = Domain()
    with pytest.raises(TypeError):
        domain == 0


def test_add_different_types():
    domain = Domain()
    with pytest.raises(TypeError):
        domain + 0


def test_str_method():
    domain = Domain()
    assert str(
        domain) == 'Domain(\n  Input Space: {  }\n  Output Space: {  }\n)'


def test_add_two_constant_parameters():
    c1 = ConstantParameter(value=1)
    c2 = ConstantParameter(value=2)
    assert CategoricalParameter(categories=[1, 2]) == c1 + c2


def test_add_discrete_to_constant():
    c = ConstantParameter(value=1)
    disc = DiscreteParameter(lower_bound=2, upper_bound=4)
    assert CategoricalParameter(categories=[1, 2, 3]) == c + disc


def test_add_continuous_to_constant_error():
    c = ConstantParameter(value=1)
    cont = ContinuousParameter(lower_bound=2, upper_bound=3)
    with pytest.raises(ValueError):
        c + cont


@pytest.mark.parametrize("bounds", [((0., 1.), (0., 1.), (0., 1.)), ([0., 1.], [0., 1.], [0., 1.]),
                                    np.array([[0., 1.], [0., 1.], [0., 1.]]), np.tile([0., 1.], (3, 1))])
def test_make_nd_continuous_domain(bounds):
    domain = make_nd_continuous_domain(bounds=bounds, dimensionality=3)
    ground_truth = Domain()
    ground_truth.add_float('x0', low=0.0, high=1.0)
    ground_truth.add_float('x1', low=0.0, high=1.0)
    ground_truth.add_float('x2', low=0.0, high=1.0)
    assert (domain.input_space == ground_truth.input_space)


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
