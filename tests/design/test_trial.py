from typing import Any, Dict, Tuple

import numpy as np
import pytest

from f3dasm.design.trial import Trial

pytestmark = pytest.mark.smoke


def test_trial_initialization(trial_data):
    dict_input, dict_output, job_number = trial_data
    trial = Trial(dict_input, dict_output, job_number)
    assert trial.input_data == dict_input
    assert trial.output_data == dict_output
    assert trial.job_number == job_number


def test_trial_to_numpy(trial_data):
    dict_input, dict_output, job_number = trial_data
    trial = Trial(dict_input, dict_output, job_number)
    input_array, output_array = trial.to_numpy()
    assert np.array_equal(input_array, np.array(list(dict_input.values())))
    assert np.array_equal(output_array, np.array(list(dict_output.values())))


def test_trial_get(trial_data):
    dict_input, dict_output, job_number = trial_data
    trial = Trial(dict_input, dict_output, job_number)
    assert trial.get('input1') == dict_input['input1']
    with pytest.raises(KeyError):
        trial.get('invalid_key')


def test_trial_get_output_space(trial_data):
    dict_input, dict_output, job_number = trial_data
    trial = Trial(dict_input, dict_output, job_number)
    with pytest.raises(KeyError):
        trial.get('output3')


def test_trial_set(trial_data):
    dict_input, dict_output, job_number = trial_data
    trial = Trial(dict_input, dict_output, job_number)
    trial.set('output3', 5)

    # Check if the output data is updated
    trial.output_data['output3'] == 5
