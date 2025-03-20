from typing import Callable

import pytest

from f3dasm import ExperimentData, create_datagenerator, create_sampler
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

    sampler = create_sampler(sampler='random', seed=42)
    experiment_data = ExperimentData(domain=domain)

    experiment_data = sampler.call(data=experiment_data, n_samples=10)
    experiment_data.set_project_dir(tmp_path, in_place=True)

    func = create_datagenerator(data_generator='ackley')

    func.arm(data=experiment_data)

    experiment_data = func.call(data=experiment_data, mode=mode)


def test_invalid_parallelization_mode():
    domain = make_nd_continuous_domain([[0, 1], [0, 1]])

    sampler = create_sampler(sampler='random', seed=42)
    experiment_data = ExperimentData(domain=domain)

    experiment_data = sampler.call(data=experiment_data, n_samples=10)
    func = create_datagenerator(data_generator='ackley')

    with pytest.raises(ValueError):
        experiment_data = func.call(data=experiment_data,
                                    mode='invalid')
