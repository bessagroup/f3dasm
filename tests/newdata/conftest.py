import numpy as np
import pytest

from f3dasm._src.experimentdata._experimental._newdata2 import _Data
from f3dasm.design import Domain


@pytest.fixture(scope="package")
def list_1():
    return {0: {'a': np.array([0.3, 5.0, 0.34]), 'b': 'd', 'c': 3},
            1: {'a': np.array([0.23, 5.0, 0.0]), 'b': 'f', 'c': 4},
            2: {'a': np.array([0.3, 5.0, 0.2]), 'b': 'c', 'c': 0}
            }


@pytest.fixture(scope="package")
def list_2():
    return {0: {'a': np.array([0.3, 0.2])},
            1: {'a': np.array([0.4, 0.3]), 'b': np.array([0.0, 1.0])}
            }


@pytest.fixture(scope="package")
def list_3():
    return {0: {'a': np.array([1.1, 0.2])},
            1: {'a': np.array([8.9, 0.3])},
            2: {'a': np.array([0.0, 0.87])}
            }


@pytest.fixture(scope="package")
def domain():
    domain = Domain()
    domain.add_float('a', 0.0, 1.0)
    domain.add_float('b', 0.0, 1.0)
    domain.add_float('c', 0.0, 1.0)
    domain.add_category('d', ['a', 'b', 'c'])
    domain.add_int('e', 0, 10)
    return domain
