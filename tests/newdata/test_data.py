from copy import deepcopy
from typing import Any, List

import numpy as np
import pandas as pd
import pytest

from f3dasm._src.experimentdata._columns import _Columns
from f3dasm._src.experimentdata._newdata import _Data, _Index
from f3dasm.design import Domain

pytestmark = pytest.mark.smoke

DataType = List[List[Any]]


def test_init(list_1: DataType):
    data = _Data(list_1)
    assert data.data == list_1
    assert data.columns.names == [0, 1, 2]
    assert data.indices.equals(pd.Index([0, 1, 2]))


def test_init_with_columns(list_1: DataType, columns_1: _Columns):
    data = _Data(list_1, columns_1)
    assert data.data == list_1
    assert data.names == ['a', 'b', 'c']


def test_init_with_columns_and_indices(
        list_1: DataType, columns_1: _Columns, indices_1: _Index):
    data = _Data(list_1, columns_1, indices_1)
    assert data.data == list_1
    assert data.names == ['a', 'b', 'c']
    assert data.indices.equals(pd.Index([3, 5, 6]))


def test__len__(list_1: DataType):
    data = _Data(list_1)
    assert len(data) == 3


def test__iter__(list_1: DataType):
    data = _Data(list_1)
    for i, row in enumerate(data):
        assert row == list_1[i]


def test__getitem__(list_1: DataType):
    data = _Data(list_1)
    assert data[0].data[0] == list_1[0]
    assert data[1].data[0] == list_1[1]
    assert data[2].data[0] == list_1[2]


def test__getitem__list(list_1: DataType):
    data = _Data(data=[[1, 2, 3], [4, 5, 6]], columns=_Columns(
        {'a': None, 'b': None, 'c': None}), index=_Index([3, 45]))
    assert data[[3, 45]].data == data.data


def test__add__(list_1: DataType, list_3: DataType):
    data_1 = _Data(list_1)
    data_2 = _Data(list_3)
    data_3 = data_1 + data_2
    assert data_3.data == list_1 + list_3
    assert data_3.columns.names == [0, 1, 2]


def test__add__empty(list_3: DataType):
    data_1 = _Data(columns=_Columns({0: None, 1: None, 2: None}))
    data_2 = _Data(list_3)
    data_3 = data_1 + data_2
    assert data_3.data == list_3
    assert data_3.columns.names == [0, 1, 2]


def test__eq__(list_1: DataType):
    data_1 = _Data(list_1)
    data_2 = _Data(list_1)
    assert data_1 == data_2


def test_repr_html(list_1: DataType):
    data = _Data(list_1)
    assert data._repr_html_() == data.to_dataframe()._repr_html_()

#                                                                    Properties
# =============================================================================


def test_names(list_1: DataType, columns_1: _Columns):
    data = _Data(list_1, columns=columns_1)
    assert data.names == ['a', 'b', 'c']


def test_names_default(list_1: DataType):
    data = _Data(list_1)
    assert data.names == [0, 1, 2]


def test_indices(list_1: DataType, indices_1: _Index):
    data = _Data(list_1, index=indices_1)
    assert data.indices.equals(pd.Index([3, 5, 6]))


def test_indices_default(list_1: DataType):
    data = _Data(list_1)
    assert data.indices.equals(pd.Index([0, 1, 2]))

#                                                      Alternative constructors
# =============================================================================


def test_from_indices():
    data = _Data.from_indices(pd.Index([0, 1]))
    assert data.indices.equals(pd.Index(([0, 1])))
    assert not data.names
    assert data.is_empty()


def test_from_domain(domain: Domain):
    data = _Data.from_domain(domain)
    assert data.indices.equals(pd.Index([]))
    assert data.names == ['a', 'b', 'c', 'd', 'e']
    assert data.is_empty()


def test_from_numpy():
    data = _Data.from_numpy(np.array([[1, 2, 3], [4, 5, 6]]))
    assert data.data == [[1, 2, 3], [4, 5, 6]]
    assert data.names == [0, 1, 2]
    assert data.indices.equals(pd.Index([0, 1]))


def test_from_dataframe():
    data = _Data.from_dataframe(pd.DataFrame([[1, 2, 3], [4, 5, 6]]))
    assert data.data == [[1, 2, 3], [4, 5, 6]]
    assert data.names == [0, 1, 2]
    assert data.indices.equals(pd.Index([0, 1]))


def test_reset():
    data = _Data.from_numpy(np.array([[1, 2, 3], [4, 5, 6]]))
    data.reset()
    assert data.data == []
    assert not data.names
    assert data.indices.equals(pd.Index([]))


def test_reset_with_domain(domain: Domain):
    data = _Data.from_numpy(np.array([[1, 2, 3], [4, 5, 6]]))
    data.reset(domain)
    assert data.data == []
    assert data.names == domain.names
    assert data.indices.equals(pd.Index([]))

#                                                                        Export
# =============================================================================


def test_to_numpy(list_1: DataType):
    data = _Data(list_1)
    data.to_numpy()


def to_dataframe(list_1: DataType):
    data = _Data(list_1)
    data.to_dataframe()
    assert data.to_dataframe().equals(pd.DataFrame(list_1))


def test_select_columns(list_1: DataType, columns_1: _Columns):
    data = _Data(data=[[1, 2, 3], [4, 5, 6]], columns=columns_1)
    new_data = data.select_columns(['a', 'c'])
    assert new_data.names == ['a', 'c']
    assert new_data.data == [[1, 3], [4, 6]]


def test_select_column(list_1: DataType, columns_1: _Columns):
    data = _Data(data=[[1, 2, 3], [4, 5, 6]], columns=columns_1)
    new_data = data.select_columns('a')
    assert new_data.names == ['a']
    assert new_data.data == [[1], [4]]


def test_add(list_2: DataType, list_3: DataType):
    data_0 = _Data(deepcopy(list_2))
    data_1 = _Data(deepcopy(list_2))
    data_2 = _Data(list_3)
    data_1.add(data_2.to_dataframe())
    assert data_1 == (data_0 + data_2)


def test_add_empty_rows():
    data = _Data(data=[[1, 2, 3], [4, 5, 6]])
    data.add_empty_rows(2)
    assert data.data == [[1, 2, 3], [4, 5, 6], [
        np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]]


def test_add_column():
    data = _Data(data=[[1, 2, 3], [4, 5, 6]])
    data.add_column('a')
    assert data.data == [[1, 2, 3, np.nan], [4, 5, 6, np.nan]]
    assert data.names == [0, 1, 2, 'a']


def test_remove():
    data = _Data(data=[[1, 2, 3], [4, 5, 6]])
    data.remove(0)
    assert data.data == [[4, 5, 6]]
    assert data.names == [0, 1, 2]


def test_remove_list():
    data = _Data(data=[[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    data.remove([0, 2])
    assert data.data == [[4, 5, 6]]
    assert data.names == [0, 1, 2]


def test_get_data_dict():
    data = _Data(data=[[1, 2, 3], [4, 5, 6]])
    assert data.get_data_dict(0) == {0: 1, 1: 2, 2: 3}


def test_set_data_all_columns():
    data = _Data(data=[[1, 2, 3], [4, 5, 6]])
    data.set_data(index=0, value=[4, 5, 6])
    assert data.data == [[4, 5, 6], [4, 5, 6]]


def test_set_data():
    data = _Data(data=[[1, 2, 3], [4, 5, 6]], columns=_Columns(
        {'a': None, 'b': None, 'c': None}))
    data.set_data(index=0, value=99, column='b')
    assert data.data == [[1, 99, 3], [4, 5, 6]]


def test_set_data_no_valid_index():
    data = _Data(data=[[1, 2, 3], [4, 5, 6]], columns=_Columns(
        {'a': None, 'b': None, 'c': None}))
    with pytest.raises(IndexError):
        data.set_data(index=2, value=99, column='b')


def test_set_data_unknown_column():
    data = _Data(data=[[1, 2, 3], [4, 5, 6]], columns=_Columns(
        {'a': None, 'b': None, 'c': None}))

    data.set_data(index=0, value=99, column='d')
    assert data.names == ['a', 'b', 'c', 'd']
    assert data.data == [[1, 2, 3, 99], [4, 5, 6, np.nan]]


def test_reset_index():
    data = _Data(data=[[1, 2, 3], [4, 5, 6]], columns=_Columns(
        {'a': None, 'b': None, 'c': None}), index=_Index([3, 45]))
    data.reset_index()
    assert data.indices.equals(pd.Index([0, 1]))


def test_is_empty():
    data = _Data(data=[[1, 2, 3], [4, 5, 6]], columns=_Columns(
        {'a': None, 'b': None, 'c': None}), index=_Index([3, 45]))
    assert not data.is_empty()
    data.reset()
    assert data.is_empty()


def test_has_columnnames():
    data = _Data(data=[[1, 2, 3], [4, 5, 6]], columns=_Columns(
        {'a': None, 'b': None, 'c': None}), index=_Index([3, 45]))
    assert not data.has_columnnames('d')
    assert data.has_columnnames('c')
    data.add_column('d')
    assert data.has_columnnames('d')


def test_set_columnnames():
    data = _Data(data=[[1, 2, 3], [4, 5, 6]], columns=_Columns(
        {'a': None, 'b': None, 'c': None}), index=_Index([3, 45]))
    data.set_columnnames(['d', 'f', 'g'])
    assert data.names == ['d', 'f', 'g']


if __name__ == "__main__":  # pragma: no cover
    pytest.main()

    # return [[np.array([0.3, 5.0, 0.34]), 'd', 3], [np.array(
    #     [0.23, 5.0, 0.0]), 'f', 4], [np.array([0.3, 5.0, 0.2]), 'c', 0]]
