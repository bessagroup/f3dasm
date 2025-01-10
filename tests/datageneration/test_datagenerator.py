from typing import Callable

import pytest

from f3dasm import ExperimentData
from f3dasm._src.datageneration.datagenerator_factory import convert_function
from f3dasm.datageneration import DataGenerator
from f3dasm.design import make_nd_continuous_domain

pytestmark = pytest.mark.smoke


def test_convert_function(
        experiment_data: ExperimentData, function_1: Callable):
    data_generator = convert_function(f=function_1, output=[
                                      'y0', 'y1'], kwargs={'s': 103})

    assert isinstance(data_generator, DataGenerator)

    experiment_data.evaluate(data_generator)


def test_convert_function2(
        experiment_data: ExperimentData, function_2: Callable):
    data_generator = convert_function(f=function_2, output=[
                                      'y0', 'y1'])

    assert isinstance(data_generator, DataGenerator)

    experiment_data.evaluate(data_generator)


@pytest.mark.parametrize("mode", ['sequential', 'parallel'])
def test_parallelization(mode, tmp_path):
    domain = make_nd_continuous_domain([[0, 1], [0, 1]])
    experiment_data = ExperimentData.from_sampling(
        sampler='random',
        domain=domain,
        n_samples=10,
        seed=42)

    experiment_data.remove_lockfile()
    experiment_data.set_project_dir(tmp_path)

    experiment_data.evaluate(data_generator='ackley',
                             mode=mode)


def test_invalid_parallelization_mode():
    domain = make_nd_continuous_domain([[0, 1], [0, 1]])
    experiment_data = ExperimentData.from_sampling(
        sampler='random',
        domain=domain,
        n_samples=10,
        seed=42)

    with pytest.raises(ValueError):
        experiment_data.evaluate(data_generator='ackley',
                                 mode='invalid')
