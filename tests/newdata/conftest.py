import numpy as np
import pytest

from f3dasm._src.experimentdata._columns import _Columns
from f3dasm._src.experimentdata._newdata import _Index
from f3dasm.design import Domain


@pytest.fixture(scope="package")
def list_1():
    return [[np.array([0.3, 5.0, 0.34]), 'd', 3], [np.array(
        [0.23, 5.0, 0.0]), 'f', 4], [np.array([0.3, 5.0, 0.2]), 'c', 0]]


@pytest.fixture(scope="package")
def columns_1():
    return _Columns({'a': None, 'b': None, 'c': None})


@pytest.fixture(scope="package")
def indices_1():
    return _Index([3, 5, 6])


@pytest.fixture(scope="package")
def list_2():
    return [[np.array([0.3, 0.2])], [np.array([0.4, 0.3])], [np.array([0.0, 1.0])]]


@pytest.fixture(scope="package")
def columns_2():
    return _Columns({'a': None})


@pytest.fixture(scope="package")
def list_3():
    return [[np.array([1.1, 0.2])], [np.array([8.9, 0.3])], [np.array([0.0, 0.87])]]


@pytest.fixture(scope="package")
def domain():
    domain = Domain()
    domain.add_float('a', 0.0, 1.0)
    domain.add_float('b', 0.0, 1.0)
    domain.add_float('c', 0.0, 1.0)
    domain.add_category('d', ['a', 'b', 'c'])
    domain.add_int('e', 0, 10)
    return domain
