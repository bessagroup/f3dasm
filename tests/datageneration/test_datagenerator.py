from typing import Callable

import pytest

from f3dasm import ExperimentData
from f3dasm._src.datageneration.datagenerator_factory import datagenerator
from f3dasm.datageneration import DataGenerator
from f3dasm.design import make_nd_continuous_domain

pytestmark = pytest.mark.smoke


def test_convert_function(
        experiment_data: ExperimentData, function_1: Callable):

    fn = datagenerator(output_names=['y0', 'y1'])(function_1)

    assert isinstance(fn, DataGenerator)

    _ = fn.call(experiment_data, s=103)


def test_convert_function2(
        experiment_data: ExperimentData, function_2: Callable):

    fn = datagenerator(output_names=['y0', 'y1'])(function_2)

    assert isinstance(fn, DataGenerator)

    _ = fn.call(experiment_data)


@pytest.mark.parametrize("mode", ['sequential', 'parallel'])
def test_parallelization(mode, tmp_path):
    domain = make_nd_continuous_domain([[0, 1], [0, 1]])
    experiment_data = ExperimentData.from_sampling(
        sampler='random',
        domain=domain,
        n_samples=10,
        seed=42)

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
