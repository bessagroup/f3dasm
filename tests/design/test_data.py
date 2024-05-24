
import numpy as np
import pandas as pd
import pytest

from f3dasm._src.experimentdata._data import _Data
from f3dasm.design import Domain

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
    # Assuming you have a Domain object named "domain"
    data = _Data.from_domain(domain)
    assert isinstance(data, _Data)
    assert isinstance(data.data, pd.DataFrame)


def test_data_reset(sample_data: _Data):
    # Assuming you have a Domain object named "domain"
    design = Domain()
    sample_data.reset(design)
    assert isinstance(sample_data.data, pd.DataFrame)
    assert len(sample_data) == 0


def test_data_remove(sample_data: _Data):
    indices = [0, 2]
    sample_data.remove(indices)
    assert len(sample_data) == 1


def test_data_add_numpy_arrays(sample_data: _Data):
    input_array = np.array([[1, 4], [2, 5]])
    df = pd.DataFrame(input_array, columns=sample_data.names)
    sample_data.add(df)
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
    sample_data.set_data(index=index, value=15,
                         column='output1')
    _column_index = sample_data.columns.iloc('output1')[0]
    assert sample_data.data.loc[index, _column_index] == 15


def test_data_to_numpy(sample_data: _Data):
    input_array = sample_data.to_numpy()
    assert isinstance(input_array, np.ndarray)
    assert input_array.shape == (
        len(sample_data), len(sample_data.data.columns))


def test_data_n_best_samples(sample_data: _Data):
    nosamples = 2
    output_names = 'input1'
    result = sample_data.n_best_samples(nosamples, output_names)
    assert isinstance(result, pd.DataFrame)
    assert len(result) == nosamples


def test_compatible_columns_add():
    # create a 4 column dataframe with random numpy values
    df = pd.DataFrame(np.random.randn(100, 4), columns=list('ABCD'))
    dg = pd.DataFrame(np.random.randn(100, 4), columns=list('ABCD'))

    f = _Data(df)
    g = _Data(dg)

    _ = f + g


def test_overwrite_data(sample_data: _Data):
    overwrite_data = _Data(pd.DataFrame(
        {'input1': [5, 6, 7], 'input2': [8, 9, 10]}))

    sample_data.overwrite(other=overwrite_data, indices=[0, 1, 2])

    pd.testing.assert_frame_equal(sample_data.data, overwrite_data.data,
                                  check_dtype=False, atol=1e-6)


def test_overwrite_data2(sample_data: _Data):
    overwrite_data = _Data(pd.DataFrame(
        {'input1': [5, 6, ], 'input2': [8, 9]}))

    sample_data.overwrite(other=overwrite_data, indices=[1, 2])

    ground_truth = _Data(pd.DataFrame(
        {'input1': [1, 5, 6], 'input2': [4, 8, 9]}))

    pd.testing.assert_frame_equal(sample_data.data, ground_truth.data,
                                  check_dtype=False, atol=1e-6)
