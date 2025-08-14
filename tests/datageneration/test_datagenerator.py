from typing import Callable

import numpy as np
import pytest

from f3dasm import (
    DataGenerator,
    ExperimentData,
    create_datagenerator,
)
from f3dasm._src.datageneration.datagenerator_factory import datagenerator
from f3dasm.design import Domain

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
    domain = Domain()
    domain.add_parameter('x')
    domain.add_output('y')
    experiment_data = ExperimentData(
        domain=domain,
        input_data=[{'x': np.array([0.5, 0.5])}])

    experiment_data.set_project_dir(tmp_path, in_place=True)

    func = create_datagenerator(data_generator='ackley', output_names='y')

    func.arm(data=experiment_data)

    experiment_data = func.call(data=experiment_data, mode=mode)


def test_invalid_parallelization_mode():
    domain = Domain()
    domain.add_parameter('x')
    domain.add_output('y')
    experiment_data = ExperimentData(
        domain=domain,
        input_data=[{'x': np.array([0.5, 0.5])}])

    func = create_datagenerator(data_generator='ackley', output_names='y')

    with pytest.raises(ValueError):
        experiment_data = func.call(data=experiment_data,
                                    mode='invalid')
