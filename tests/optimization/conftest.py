import numpy as np
import pytest

from f3dasm import ExperimentData
from f3dasm.design import Domain

DIM = 2


@pytest.fixture(scope="package")
def data():
    domain = Domain()
    domain.add_array('x', shape=DIM, low=0.0, high=1.0)
    domain.add_output('y')

    x0 = np.array([0.5]*DIM)
    return ExperimentData(
        domain=domain,
        input_data=[{'x': x0}],
    )


@pytest.fixture(scope="package")
def data2():
    domain = Domain()
    domain.add_float('x0', low=0.0, high=1.0)
    domain.add_output('y')

    return ExperimentData(
        domain=domain,
        input_data=[{'x0': 0.5}],
    )
