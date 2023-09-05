from io import TextIOWrapper
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from f3dasm.design._data import _Data
from f3dasm.design.design import Design
from f3dasm.design.domain import Domain

pytestmark = pytest.mark.smoke


@pytest.fixture
def sample_data():
    input_data = pd.DataFrame({'input1': [1, 2, 3], 'input2': [4, 5, 6]})
    return _Data(input_data)


def test_data_initialization(sample_data: _Data):
    assert isinstance(sample_data.data, pd.DataFrame)


def test_data_len(sample_data: _Data):
    assert len(sample_data) == len(sample_data.data)


def test_data_repr_html(sample_data: _Data):
    assert isinstance(sample_data._repr_html_(), str)


def test_data_from_design(domain: Domain):
    # Assuming you have a DesignSpace object named "design"
    data = _Data.from_domain(domain)
    assert isinstance(data, _Data)
    assert isinstance(data.data, pd.DataFrame)


def test_data_reset(sample_data: _Data):
    # Assuming you have a DesignSpace object named "design"
    design = Domain()
    sample_data.reset(design)
    assert isinstance(sample_data.data, pd.DataFrame)
    assert len(sample_data) == 0


def test_data_select(sample_data: _Data):
    indices = [0, 2]
    sample_data.select(indices)
    assert len(sample_data) == len(indices)


def test_data_remove(sample_data: _Data):
    indices = [0, 2]
    sample_data.remove(indices)
    assert len(sample_data) == 1


def test_data_add_numpy_arrays(sample_data: _Data):
    input_array = np.array([[1, 4], [2, 5]])
    sample_data.add_numpy_arrays(input_array)
    assert len(sample_data) == 5


def test_data_get_data(sample_data: _Data):
    input_data = sample_data.data
    assert isinstance(input_data, pd.DataFrame)
    assert input_data.equals(sample_data.data)


def test_data_get_inputdata_dict(sample_data: _Data):
    index = 0
    input_dict = sample_data.get_data_dict(index)
    assert isinstance(input_dict, dict)
    assert input_dict == {'input1': 1, 'input2': 4}


def test_data_set_data(sample_data: _Data):
    index = 0
    sample_data.set_data(index, 15, 'output1')
    assert sample_data.data.loc[index, 'output1'] == 15


def test_data_to_numpy(sample_data: _Data):
    input_array = sample_data.to_numpy()
    assert isinstance(input_array, np.ndarray)
    assert input_array.shape == (len(sample_data), len(sample_data.data.columns))


def test_data_n_best_samples(sample_data: _Data):
    nosamples = 2
    output_names = 'input1'
    result = sample_data.n_best_samples(nosamples, output_names)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == nosamples
