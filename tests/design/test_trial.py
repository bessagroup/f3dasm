from typing import Any, Dict, Tuple

import numpy as np
import pytest

from f3dasm.design.design import Design

pytestmark = pytest.mark.smoke


def test_design_initialization(design_data):
    dict_input, dict_output, job_number = design_data
    design = Design(dict_input, dict_output, job_number)
    assert design.input_data == dict_input
    assert design.output_data == dict_output
    assert design.job_number == job_number


def test_design_to_numpy(design_data):
    dict_input, dict_output, job_number = design_data
    design = Design(dict_input, dict_output, job_number)
    input_array, output_array = design.to_numpy()
    assert np.array_equal(input_array, np.array(list(dict_input.values())))
    assert np.array_equal(output_array, np.array(list(dict_output.values())))


def test_design_get(design_data):
    dict_input, dict_output, job_number = design_data
    design = Design(dict_input, dict_output, job_number)
    assert design.get('input1') == dict_input['input1']
    with pytest.raises(KeyError):
        design.get('invalid_key')


def test_design_get_output_space(design_data):
    dict_input, dict_output, job_number = design_data
    design = Design(dict_input, dict_output, job_number)
    with pytest.raises(KeyError):
        design.get('output3')


def test_design_set(design_data):
    dict_input, dict_output, job_number = design_data
    design = Design(dict_input, dict_output, job_number)
    design.set('output3', 5)

    # Check if the output data is updated
    design.output_data['output3'] == 5
