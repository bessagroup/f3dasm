from typing import Callable

import pytest

from f3dasm import ExperimentData
from f3dasm.datageneration import DataGenerator, convert_function

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
