from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from f3dasm._src.experimentdata._experimental._newdata2 import (
    _convert_dict_to_data, _Data, _data_factory)

pytestmark = pytest.mark.smoke

DataType = Dict[int, Dict[str, Any]]

#                                                                Initialization
# =============================================================================


def test_init():
    data = _Data({0: {"a": 1, "b": 2}})
    assert len(data) == 1
    assert not data.is_empty()
    assert data.data == {0: {"a": 1, "b": 2}}


def test_init_empty():
    data = _Data()
    assert len(data) == 0
    assert data.is_empty()


def test_init_with_data():
    input_data = {0: {"a": 1, "b": 2}}
    data = _Data(input_data)
    assert len(data) == 1
    assert not data.is_empty()
    assert data.data == input_data


def test_from_numpy():
    array = np.array([[1, 2, 3], [4, 5, 6]])
    data = _Data.from_numpy(array)
    expected_data = {0: {0: 1, 1: 2, 2: 3}, 1: {0: 4, 1: 5, 2: 6}}
    assert data.data == expected_data


def test_from_numpy_with_keys():
    array = np.array([[1, 2, 3], [4, 5, 6]])
    data = _Data.from_numpy(array, keys=["a", "b", "c"])
    expected_data = {0: {"a": 1, "b": 2, "c": 3}, 1: {"a": 4, "b": 5, "c": 6}}
    assert data.data == expected_data


def test_from_dataframe():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    data = _Data.from_dataframe(df)
    expected_data = {0: {"a": 1, "b": 3}, 1: {"a": 2, "b": 4}}
    assert data.data == expected_data


def test_from_indices():
    data = _Data.from_indices([0, 1])
    assert data.data == {0: {}, 1: {}}

#                                                                     Exporting
# =============================================================================


def test_to_numpy():
    input_data = {0: {"a": 1, "b": 2}, 1: {"a": 3, "b": 4}}
    data = _Data(input_data)
    np_array = data.to_numpy()
    expected_array = np.array([[1, 2], [3, 4]])
    np.testing.assert_array_equal(np_array, expected_array)


def test_to_dataframe():
    input_data = {0: {"a": 1, "b": 2}, 1: {"a": 3, "b": 4}}
    data = _Data(input_data)
    df = data.to_dataframe()
    expected_df = pd.DataFrame({"a": [1, 3], "b": [2, 4]})
    pd.testing.assert_frame_equal(df, expected_df)


def test_to_xarray():
    input_data = {0: {"a": 1, "b": 2}, 1: {"a": 3, "b": 4}}
    data = _Data(input_data)
    xarray = data.to_xarray('test')
    expected_xarray = xr.DataArray(
        [[1, 2], [3, 4]], dims=["iterations", "test"],
        coords={"iterations": [0, 1], "test": ["a", "b"]})
    xr.testing.assert_equal(xarray, expected_xarray)


def test_get_data_dict():
    input_data = {0: {"a": 1, "b": 2}, 1: {"a": 3, "b": 4}}
    data = _Data(input_data)
    assert data.get_data_dict(0) == {"a": 1, "b": 2}


def test_convert_dict_to_data():
    dictionary = {"a": 1, "b": 2}
    data = _convert_dict_to_data(dictionary)
    expected_data = _Data({0: {"a": 1, "b": 2}})
    assert data == expected_data

#                                                                    Properties
# =============================================================================


def test_len():
    data = _Data({0: {"a": 1}, 1: {"a": 2}})
    assert len(data) == 2


def test_indices():
    data = _Data({0: {"a": 1}, 1: {"a": 2}})
    assert data.indices == [0, 1]


def test_names():
    data = _Data({0: {"a": 1}, 1: {"a": 2}})
    assert data.names == ["a"]


def test_is_empty():
    data = _Data()
    assert data.is_empty()
    data = _Data({0: {"a": 1}})
    assert not data.is_empty()


def test_getitem():
    data = _Data({0: {"a": 1}, 1: {"a": 2}})
    assert data[0] == _Data({0: {"a": 1}})
    assert data[1] == _Data({1: {"a": 2}})
    assert data[[0, 1]] == data


def test_repr():
    data = _Data({0: {"a": 1}, 1: {"a": 2}})
    assert isinstance(data.__repr__(), str)


def test_repr_html():
    data = _Data({0: {"a": 1}, 1: {"a": 2}})
    assert isinstance(data._repr_html_(), str)
#                                                       Selecting and combining
# =============================================================================


def test_join():
    data1 = _Data({0: {"a": 1, "b": 2}, 1: {"a": 3, "b": 4}})
    data2 = _Data({0: {"c": 5, "d": 6}, 1: {"c": 7, "d": 8}})
    data3 = data1.join(data2)
    expected_data = {0: {"a": 1, "b": 2, "c": 5, "d": 6},
                     1: {"a": 3, "b": 4, "c": 7, "d": 8}}
    assert data3 == _Data(expected_data)


def test_select_columns():
    input_data = {0: {"a": 1, "b": 2, "c": 3}, 1: {"a": 4, "b": 5, "c": 6}}
    data = _Data(input_data)
    selected_data = data.select_columns(["a", "c"])
    expected_data = {0: {"a": 1, "c": 3}, 1: {"a": 4, "c": 6}}
    assert selected_data.data == expected_data


def test_select_columns_single():
    input_data = {0: {"a": 1, "b": 2, "c": 3}, 1: {"a": 4, "b": 5, "c": 6}}
    data = _Data(input_data)
    selected_data = data.select_columns("a")
    expected_data = {0: {"a": 1}, 1: {"a": 4}}
    assert selected_data.data == expected_data


def test_rename_columns():
    input_data = {0: {"a": 1, "b": 2, "c": 3}, 1: {"a": 4, "b": 5, "c": 6}}
    data = _Data(input_data)
    data.rename_columns({"a": "x", "b": "y"})
    expected_data = {0: {"x": 1, "y": 2, "c": 3}, 1: {"x": 4, "y": 5, "c": 6}}
    assert data.data == expected_data


def test_drop():
    input_data = {0: {"a": 1, "b": 2, "c": 3}, 1: {"a": 4, "b": 5, "c": 6}}
    data = _Data(input_data)
    data.drop(["b"])
    expected_data = {0: {"a": 1, "c": 3}, 1: {"a": 4, "c": 6}}
    assert data.data == expected_data


def test_drop_single_key():
    input_data = {0: {"a": 1, "b": 2, "c": 3}, 1: {"a": 4, "b": 5, "c": 6}}
    data = _Data(input_data)
    data.drop("b")
    expected_data = {0: {"a": 1, "c": 3}, 1: {"a": 4, "c": 6}}
    assert data.data == expected_data

#                                                                     Modifying
# =============================================================================


def test_add():
    data1 = _Data({0: {"a": 1, "b": 2}})
    data2 = _Data({0: {"a": 3, "b": 4}})
    data3 = data1 + data2
    expected_data = {0: {"a": 1, "b": 2}, 1: {"a": 3, "b": 4}}
    assert data3.data == expected_data


def test_add_empty():
    data1 = _Data()
    data2 = _Data({0: {"a": 3, "b": 4}})
    data3 = data1 + data2
    assert data3.data == {0: {"a": 3, "b": 4}}


def test_add_column():
    missing_value = np.nan
    data = _Data({0: {"a": 1}, 1: {"a": 2}})
    data.add_column("b")
    expected_data = {0: {"a": 1, "b": missing_value},
                     1: {"a": 2, "b": missing_value}}
    assert data.data == expected_data


def test_overwrite():
    data = _Data({0: {"a": 1, "b": 2}, 1: {"a": 3, "b": 4}})
    data2 = _Data({0: {"a": 5, "b": 6}, 1: {"a": 7, "b": 8}})
    data.overwrite([0], data2)
    assert data.data == {0: {"a": 5, "b": 6}, 1: {"a": 3, "b": 4}}


def test_remove():
    data = _Data({0: {"a": 1, "b": 2}, 1: {"a": 3, "b": 4}})
    data.remove([1])
    assert data.data == {0: {"a": 1, "b": 2}}


def test_n_best_samples():
    df = pd.DataFrame({"a": [3, 1, 2], "b": [6, 4, 5]})
    data = _Data.from_dataframe(df)
    best_samples = data.n_best_samples(2, "a")
    expected_df = pd.DataFrame({"a": [1, 2], "b": [4, 5]}, index=[1, 2])
    pd.testing.assert_frame_equal(best_samples, expected_df)


def test_set_data():
    data = _Data({0: {"a": 1}})
    data.set_data(0, 2, "a")
    assert data.data[0]["a"] == 2


def test_reset_index():
    data = _Data({1: {"a": 1}, 3: {"a": 2}})
    data.reset_index()
    expected_data = {0: {"a": 1}, 1: {"a": 2}}
    assert data.data == expected_data


def test_data_factory_pandas():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    data = _data_factory(df)
    expected_data = _Data.from_dataframe(df)
    assert data == expected_data


def test_data_factory_numpy():
    np_array = np.array([[1, 2], [3, 4]])
    data = _data_factory(np_array)
    expected_data = _Data.from_numpy(np_array)
    assert data == expected_data


def test_data_factory_none():
    data = _data_factory(None)
    expected_data = _Data()
    assert data == expected_data


def test_data_factory_unrecognized_datatype():
    with pytest.raises(TypeError):
        _ = _data_factory(0)


def test_data_factory_data_object():
    data = _data_factory(_Data({0: {"a": 1}}))
    expected_data = _Data({0: {"a": 1}})
    assert data == expected_data


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
