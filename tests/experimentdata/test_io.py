"""Tests for I/O utility functions in f3dasm._src._io."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from f3dasm._src._io import (
    _project_dir_factory,
    copy_object,
    load_object,
    numpy_load,
    numpy_store,
    pandas_load,
    pandas_store,
    pickle_load,
    pickle_store,
    store_object,
    xarray_dataarray_load,
    xarray_dataarray_store,
    xarray_dataset_load,
    xarray_dataset_store,
)

pytestmark = pytest.mark.smoke


# ======================= pickle store/load =======================


def test_pickle_store_load_roundtrip(tmp_path):
    obj = {"key": "value", "number": 42}
    path = str(tmp_path / "test_obj")
    stored_path = pickle_store(obj, path)
    loaded = pickle_load(stored_path)
    assert loaded == obj


# ======================= numpy store/load =======================


def test_numpy_store_load_roundtrip(tmp_path):
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    path = str(tmp_path / "test_arr")
    stored_path = numpy_store(arr, path)
    loaded = numpy_load(stored_path)
    np.testing.assert_array_equal(loaded, arr)


# ======================= pandas store/load =======================


def test_pandas_store_load_roundtrip(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    path = str(tmp_path / "test_df")
    stored_path = pandas_store(df, path)
    loaded = pandas_load(stored_path)
    pd.testing.assert_frame_equal(loaded, df)


# ======================= xarray store/load =======================


def test_xarray_dataset_store_load_roundtrip(tmp_path):
    ds = xr.Dataset({"var": (["x"], [1, 2, 3])})
    path = str(tmp_path / "test_ds")
    stored_path = xarray_dataset_store(ds, path)
    loaded = xarray_dataset_load(stored_path)
    xr.testing.assert_equal(loaded, ds)


def test_xarray_dataarray_store_load_roundtrip(tmp_path):
    da = xr.DataArray([1, 2, 3], dims=["x"])
    path = str(tmp_path / "test_da")
    stored_path = xarray_dataarray_store(da, path)
    loaded = xarray_dataarray_load(stored_path)
    xr.testing.assert_equal(loaded, da)


# ======================= _project_dir_factory =======================


def test_project_dir_factory_path(tmp_path):
    result = _project_dir_factory(tmp_path)
    assert result == tmp_path.absolute()


def test_project_dir_factory_string(tmp_path):
    result = _project_dir_factory(str(tmp_path))
    assert result.is_absolute()


def test_project_dir_factory_none():
    from pathlib import Path

    result = _project_dir_factory(None)
    assert result == Path.cwd()


def test_project_dir_factory_invalid_type():
    with pytest.raises(TypeError):
        _project_dir_factory(123)


# ======================= store_object =======================


def test_store_object_numpy(tmp_path):
    arr = np.array([1, 2, 3])
    path = store_object(project_dir=tmp_path, object=arr, name="test", id=0)
    assert path.endswith(".npy")


def test_store_object_unknown_type_falls_back_to_pickle(tmp_path):
    obj = {"custom": True}
    path = store_object(project_dir=tmp_path, object=obj, name="test", id=0)
    assert path.endswith(".pkl")


def test_store_object_with_custom_function(tmp_path):
    obj = "test_data"

    def custom_store(obj, path):
        from pathlib import Path

        p = Path(path).with_suffix(".txt")
        p.write_text(obj)
        return str(p)

    path = store_object(
        project_dir=tmp_path,
        object=obj,
        name="test",
        id=0,
        store_function=custom_store,
    )
    assert path.endswith(".txt")


# ======================= load_object =======================


def test_load_object_numpy(tmp_path):
    arr = np.array([1, 2, 3])
    rel_path = store_object(
        project_dir=tmp_path, object=arr, name="test", id=0
    )
    loaded = load_object(project_dir=tmp_path, path=rel_path)
    np.testing.assert_array_equal(loaded, arr)


def test_load_object_unknown_suffix_uses_pickle(tmp_path):
    obj = [1, 2, 3]
    rel_path = store_object(
        project_dir=tmp_path, object=obj, name="test", id=0
    )
    loaded = load_object(project_dir=tmp_path, path=rel_path)
    assert loaded == obj


# ======================= copy_object =======================


def test_copy_object(tmp_path):
    from pathlib import Path

    old_dir = tmp_path / "old"
    new_dir = tmp_path / "new"

    arr = np.array([1, 2, 3])
    rel_path = store_object(project_dir=old_dir, object=arr, name="test", id=0)
    copied_path = copy_object(
        object_path=Path(rel_path),
        old_project_dir=old_dir,
        new_project_dir=new_dir,
    )
    loaded = load_object(project_dir=new_dir, path=copied_path)
    np.testing.assert_array_equal(loaded, arr)
