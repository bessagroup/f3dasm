from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from f3dasm import ExperimentData
from f3dasm._src._io import ReferenceValue
from f3dasm.design import Domain

pytestmark = pytest.mark.smoke


class CustomObject:
    @classmethod
    def load(cls, _path):
        return CustomObject()

    def save(self, _path):
        ...


def custom_object_store(object: CustomObject, path: str) -> str:
    _path = Path(path).with_suffix(".xxx")
    object.save(_path)
    return str(_path)


def custom_object_load(path: str) -> CustomObject:
    _path = Path(path).with_suffix(".xxx")
    return CustomObject.load(_path)


@pytest.fixture
def domain_with_custom_object() -> Domain:
    domain = Domain()
    domain.add_float(name="x1", low=0, high=1)
    domain.add_parameter(name="custom_in", store_function=custom_object_store,
                         load_function=custom_object_load, to_disk=True)
    domain.add_output(name="y")
    domain.add_output(name="custom_out", store_function=custom_object_store,
                      load_function=custom_object_load, to_disk=True)
    return domain


def test_custom_object(domain_with_custom_object, tmp_path):
    input_data = [{'custom_in': CustomObject(), 'x1': 0.5}]
    output_data = [{'y': 1, 'custom_out': CustomObject()}]
    data = ExperimentData(domain=domain_with_custom_object, input_data=input_data,
                          output_data=output_data, project_dir=tmp_path)

    assert isinstance(data.data[0].input_data['custom_in'], CustomObject)
    assert isinstance(data.data[0].output_data['custom_out'], CustomObject)
    assert isinstance(data.data[0]._input_data['custom_in'], ReferenceValue)


def test_numpy_array_object(tmp_path):
    domain = Domain()
    domain.add_parameter('x', to_disk=True)
    x = np.array([1, 2, 3])
    input_data = [{'x': x}]
    data = ExperimentData(
        domain=domain, input_data=input_data, project_dir=tmp_path)

    assert isinstance(data.data[0].input_data['x'], np.ndarray)
    assert isinstance(data.data[0]._input_data['x'], ReferenceValue)
    assert np.allclose(data.data[0].input_data['x'], x)


def test_pandas_dataframe_object(tmp_path):
    domain = Domain()
    domain.add_parameter('x', to_disk=True)
    x = pd.DataFrame([{f'u{i}': 3.2 for i in range(10)} for _ in range(10)])
    input_data = [{'x': x}]
    data = ExperimentData(
        domain=domain, input_data=input_data, project_dir=tmp_path)

    assert isinstance(data.data[0].input_data['x'], pd.DataFrame)
    assert isinstance(data.data[0]._input_data['x'], ReferenceValue)
    pd.testing.assert_frame_equal(data.data[0].input_data['x'], x)


def test_xarray_object(tmp_path):
    domain = Domain()
    domain.add_parameter('x', to_disk=True)
    x = pd.DataFrame([{f'u{i}': 3.2 for i in range(10)} for _ in range(10)])
    input_data = [{'x': x}]
    data = ExperimentData(
        domain=domain, input_data=input_data, project_dir=tmp_path)

    assert isinstance(data.data[0].input_data['x'], pd.DataFrame)
    assert isinstance(data.data[0]._input_data['x'], ReferenceValue)
    pd.testing.assert_frame_equal(data.data[0].input_data['x'], x)


def test_xarray_dataarray_object(tmp_path):
    domain = Domain()
    domain.add_parameter('x', to_disk=True)

    # Create an xarray.DataArray
    x = xr.DataArray(
        data=[[3.2 for i in range(10)] for _ in range(10)],
        dims=["row", "col"],
        coords={"row": range(10), "col": [f'u{i}' for i in range(10)]},
        name="x_data"
    )

    input_data = [{'x': x}]
    data = ExperimentData(
        domain=domain, input_data=input_data, project_dir=tmp_path)

    # Assertions
    assert isinstance(data.data[0].input_data['x'], xr.DataArray)
    assert isinstance(data.data[0]._input_data['x'], ReferenceValue)
    xr.testing.assert_equal(data.data[0].input_data['x'], x)


def test_xarray_dataset_object(tmp_path):
    domain = Domain()
    domain.add_parameter('x', to_disk=True)

    # Create an xarray.DataArray
    _x = xr.DataArray(
        data=[[3.2 for i in range(10)] for _ in range(10)],
        dims=["row", "col"],
        coords={"row": range(10), "col": [f'u{i}' for i in range(10)]},
    )

    x = xr.Dataset({'x': _x})
    input_data = [{'x': x}]
    data = ExperimentData(
        domain=domain, input_data=input_data, project_dir=tmp_path)

    # Assertions
    assert isinstance(data.data[0].input_data['x'], xr.Dataset)
    assert isinstance(data.data[0]._input_data['x'], ReferenceValue)
    xr.testing.assert_equal(data.data[0].input_data['x'], x)


def test_xarray_pickle_object(tmp_path):
    domain = Domain()
    domain.add_parameter('x', to_disk=True)

    # Create an xarray.DataArray
    x = CustomObject()
    input_data = [{'x': x}]
    data = ExperimentData(
        domain=domain, input_data=input_data, project_dir=tmp_path)

    # Assertions
    assert isinstance(data.data[0].input_data['x'], CustomObject)
    assert isinstance(data.data[0]._input_data['x'], ReferenceValue)
