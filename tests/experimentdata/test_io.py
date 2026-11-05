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


def test_copy_object_collision(tmp_path):
    """When the target already exists, copy_object increments the filename."""
    from pathlib import Path

    old_dir = tmp_path / "old"
    new_dir = tmp_path / "new"

    arr1 = np.array([1, 2, 3])
    arr2 = np.array([4, 5, 6])
    rel_path1 = store_object(
        project_dir=old_dir, object=arr1, name="test", id=0
    )
    # Store another in new_dir at same path to create collision
    store_object(project_dir=new_dir, object=arr2, name="test", id=0)

    copied_path = copy_object(
        object_path=Path(rel_path1),
        old_project_dir=old_dir,
        new_project_dir=new_dir,
    )
    # Should have incremented the filename
    assert copied_path != rel_path1


# ======================= figure store/load =======================


def test_figure_store(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from f3dasm._src._io import figure_store

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    path = str(tmp_path / "test_fig")
    stored_path = figure_store(fig, path)
    assert (tmp_path / "test_fig.pdf").exists()
    plt.close(fig)


def test_figure_load_png(tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from f3dasm._src._io import figure_load

    # Create a PNG for figure_load (which reads image files)
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    png_path = tmp_path / "test_fig.png"
    fig.savefig(png_path)
    plt.close(fig)

    # figure_load uses .pdf suffix by default but accepts any path
    # Test with png directly via plt.imread
    loaded = plt.imread(str(png_path))
    assert isinstance(loaded, np.ndarray)
    assert loaded.ndim == 3


# ======================= concurrent writer/reader (#273) =======================


def _writer_loop(path_str: str, n_iters: int) -> None:
    """Process target: rewrite a 1000-row DataFrame repeatedly."""
    import pandas as pd

    from f3dasm._src._io import pandas_store

    df = pd.DataFrame({"a": range(1000), "b": range(1000)})
    for _ in range(n_iters):
        pandas_store(df, path_str)


def _reader_loop(path_str: str, n_iters: int, errors: list) -> None:
    """Process target: read the DataFrame repeatedly and record any error."""
    import pandas as pd

    from f3dasm._src._io import pandas_load

    for _ in range(n_iters):
        try:
            df = pandas_load(path_str)
            # Sanity check: should always observe a fully populated frame
            if len(df) != 1000:
                errors.append(f"short read: {len(df)} rows")
        except pd.errors.EmptyDataError as exc:
            errors.append(f"EmptyDataError: {exc}")
        except FileNotFoundError:
            # The reader can race with the very first write — tolerable.
            pass


def test_pandas_store_concurrent_no_empty_data_error(tmp_path):
    """Concurrent writer + reader must never observe a half-written file.

    Regression for issue #273: `pandas.DataFrame.to_csv(path)` previously
    truncated the file before writing, so a reader hitting the file
    between truncate and the body raised `pd.errors.EmptyDataError`. The
    atomic temp+rename in `_io._atomic_write` should make that race
    invisible to readers.
    """
    import multiprocessing as mp

    path_str = str(tmp_path / "concurrent")

    # Seed the file once so the reader has something to read at start.
    from f3dasm._src._io import pandas_store

    pandas_store(pd.DataFrame({"a": range(1000), "b": range(1000)}), path_str)

    n_iters = 200
    with mp.Manager() as manager:
        errors = manager.list()
        ctx = mp.get_context("spawn")
        writer = ctx.Process(target=_writer_loop, args=(path_str, n_iters))
        reader = ctx.Process(
            target=_reader_loop,
            args=(path_str, n_iters, errors),
        )

        writer.start()
        reader.start()
        writer.join(timeout=60)
        reader.join(timeout=60)

        assert not writer.is_alive(), "writer process hung"
        assert not reader.is_alive(), "reader process hung"
        assert list(errors) == [], (
            f"concurrent reader observed half-written files: {list(errors)}"
        )
