from typing import Callable

import pytest

from f3dasm import ExperimentData, create_sampler
from f3dasm.design import Domain


@pytest.fixture(scope="package")
def experiment_data() -> ExperimentData:
    domain = Domain()
    domain.add_float('x', low=0.0, high=1.0)

    experiment_data = ExperimentData(domain=domain)

    sampler = create_sampler(sampler='random', seed=2023)

    experiment_data = sampler.call(data=experiment_data, n_samples=100)

    return experiment_data


def example_function(x: int, s: int):
    return x + s, x - s


def example_function2(x: int):
    return x, -x


@pytest.fixture(scope="package")
def function_1() -> Callable:
    return example_function


@pytest.fixture(scope="package")
def function_2() -> Callable:
    return example_function2
