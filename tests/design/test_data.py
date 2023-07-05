from io import TextIOWrapper
from typing import Any, Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from f3dasm.design._data import Trial, _Data
from f3dasm.design.design import DesignSpace

pytestmark = pytest.mark.smoke

@pytest.fixture
def sample_data():
    input_data = pd.DataFrame({'input1': [1, 2, 3], 'input2': [4, 5, 6]})
    output_data = pd.DataFrame({'output1': [7, 8, 9], 'output2': [10, 11, 12]})
    data = pd.concat([input_data, output_data], axis=1, keys=['input', 'output'])
    return _Data(data)


def test_data_initialization(sample_data):
    assert isinstance(sample_data.data, pd.DataFrame)


def test_data_len(sample_data):
    assert len(sample_data) == len(sample_data.data)


def test_data_repr_html(sample_data):
    assert isinstance(sample_data._repr_html_(), str)


def test_data_from_design(design_space):
    # Assuming you have a DesignSpace object named "design"
    data = _Data.from_design(design_space)
    assert isinstance(data, _Data)
    assert isinstance(data.data, pd.DataFrame)


def test_data_reset(sample_data):
    # Assuming you have a DesignSpace object named "design"
    design = DesignSpace()
    sample_data.reset(design)
    assert isinstance(sample_data.data, pd.DataFrame)
    assert len(sample_data) == 0


def test_data_select(sample_data):
    indices = [0, 2]
    sample_data.select(indices)
    assert len(sample_data) == len(indices)


def test_data_remove(sample_data):
    indices = [0, 2]
    sample_data.remove(indices)
    assert len(sample_data) == 1


def test_data_add_numpy_arrays(sample_data):
    input_array = np.array([[1, 4], [2, 5]])
    output_array = np.array([[7, 10], [8, 11]])
    sample_data.add_numpy_arrays(input_array, output_array)
    assert len(sample_data) == 5


def test_data_get_inputdata(sample_data):
    input_data = sample_data.get_inputdata()
    assert isinstance(input_data, pd.DataFrame)
    assert input_data.equals(sample_data.data['input'])


def test_data_get_outputdata(sample_data):
    output_data = sample_data.get_outputdata()
    assert isinstance(output_data, pd.DataFrame)
    assert output_data.equals(sample_data.data['output'])


def test_data_get_inputdata_dict(sample_data):
    index = 0
    input_dict = sample_data.get_inputdata_dict(index)
    assert isinstance(input_dict, dict)
    assert input_dict == {'input1': 1, 'input2': 4}


def test_data_get_outputdata_dict(sample_data):
    index = 0
    output_dict = sample_data.get_outputdata_dict(index)
    assert isinstance(output_dict, dict)
    assert output_dict == {'output1': 7, 'output2': 10}


def test_data_get_trial(sample_data):
    index = 0
    trial = sample_data.get_trial(index)
    assert isinstance(trial, Trial)
    assert trial.job_number == index


def test_data_set_trial(sample_data):
    trial = Trial({'input1': 1, 'input2': 4}, {'output1': 7, 'output2': 10}, 0)
    sample_data.set_trial(trial)
    assert sample_data.data.loc[0, ('output', 'output1')] == 7
    assert sample_data.data.loc[0, ('output', 'output2')] == 10


def test_data_set_outputdata(sample_data):
    index = 0
    sample_data.set_outputdata(index, 15)
    assert sample_data.data.loc[index, ('output', 'output1')] == 15


def test_data_to_numpy(sample_data):
    input_array, output_array = sample_data.to_numpy()
    assert isinstance(input_array, np.ndarray)
    assert isinstance(output_array, np.ndarray)
    assert input_array.shape == (len(sample_data), len(sample_data.data['input'].columns))
    assert output_array.shape == (len(sample_data), len(sample_data.data['output'].columns))


def test_data_n_best_samples(sample_data):
    nosamples = 2
    output_names = ['output1']
    result = sample_data.n_best_samples(nosamples, output_names)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == nosamples


def test_data_number_of_datapoints(sample_data):
    assert sample_data.number_of_datapoints() == len(sample_data)
