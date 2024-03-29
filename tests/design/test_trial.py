import numpy as np
import pytest

from f3dasm import ExperimentSample

pytestmark = pytest.mark.smoke


def test_design_initialization(design_data):
    dict_input, dict_output, job_number = design_data
    design = ExperimentSample(dict_input, dict_output, job_number)
    assert design.input_data == dict_input
    assert design._dict_output == dict_output
    assert design.job_number == job_number


def test_design_to_numpy(design_data):
    dict_input, dict_output, job_number = design_data
    design = ExperimentSample(dict_input, dict_output, job_number)
    input_array, output_array = design.to_numpy()

    check_output_array = np.array([v for v, _ in dict_output.values()])
    assert np.array_equal(output_array, check_output_array)


def test_design_set(design_data):
    dict_input, dict_output, job_number = design_data
    design = ExperimentSample(dict_input, dict_output, job_number)
    design['output3'] = 5

    # Check if the output data is updated
    design.output_data['output3'] == 5
