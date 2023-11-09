import pandas as pd
import pytest

from f3dasm.design import (Domain, _CategoricalParameter, _ContinuousParameter,
                           _DiscreteParameter)


@pytest.fixture(scope="package")
def doe():
    x1 = _ContinuousParameter(lower_bound=2.4, upper_bound=10.3)
    x2 = _DiscreteParameter(lower_bound=5, upper_bound=80)
    x3 = _ContinuousParameter(lower_bound=10.0, upper_bound=380.3)
    x4 = _CategoricalParameter(categories=["test1", "test2", "test3"])
    x5 = _DiscreteParameter(lower_bound=2, upper_bound=3)

    designspace = {'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4, 'x5': x5}

    doe = Domain(space=designspace)
    return doe


@pytest.fixture(scope="package")
def domain():

    space = {
        'x1': _ContinuousParameter(-5.12, 5.12),
        'x2': _DiscreteParameter(-3, 3),
        'x3': _CategoricalParameter(["red", "green", "blue"])
    }

    return Domain(space=space)


@pytest.fixture(scope="package")
def continuous_parameter():
    lower_bound = 3.3
    upper_bound = 3.8
    return _ContinuousParameter(lower_bound=lower_bound,
                                upper_bound=upper_bound)


@pytest.fixture(scope="package")
def discrete_parameter():
    lower_bound = 3
    upper_bound = 6
    return _DiscreteParameter(lower_bound=lower_bound, upper_bound=upper_bound)


@pytest.fixture(scope="package")
def categorical_parameter():
    categories = ["test1", "test2", "test3"]
    return _CategoricalParameter(categories=categories)


@pytest.fixture(scope="package")
def design_data():
    dict_input = {'input1': 1, 'input2': 2}
    dict_output = {'output1': (3, False), 'output2': (4, False)}
    job_number = 123
    return dict_input, dict_output, job_number


@pytest.fixture(scope="package")
def sample_dataframe():
    data = {
        'feature1': [1.0, 2.0, 3.0],
        'feature2': [4, 5, 6],
        'feature3': ['A', 'B', 'C']
    }
    return pd.DataFrame(data)
