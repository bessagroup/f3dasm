from __future__ import annotations

from copy import copy
from typing import Iterable

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from f3dasm.design import ContinuousParameter, Domain, ExperimentData, Status
from f3dasm.sampling import Sampler  # Import your Sampler class
from f3dasm.sampling import RandomUniform

pytestmark = pytest.mark.smoke


def test_check_experimentdata(experimentdata: ExperimentData):
    assert isinstance(experimentdata, ExperimentData)

# Write test functions


def test_experiment_data_init(experimentdata: ExperimentData, domain: Domain):
    assert experimentdata.domain == domain
    assert experimentdata.filename == 'experimentdata'
    # Add more assertions as needed


def test_experiment_data_add(experimentdata: ExperimentData,
                             experimentdata2: ExperimentData, domain: Domain):
    experimentdata_total = ExperimentData(domain)
    experimentdata_total.add_experiments(experimentdata)
    experimentdata_total.add_experiments(experimentdata2)
    assert experimentdata_total == experimentdata + experimentdata2


def test_experiment_data_len_empty(domain: Domain):
    experiment_data = ExperimentData(domain)
    assert len(experiment_data) == 0  # Update with the expected length


def test_experiment_data_len_equals_input_data(experimentdata: ExperimentData):
    assert len(experimentdata) == len(experimentdata.input_data)


def test_experiment_data_len_equals_output_data(experimentdata: ExperimentData):
    assert len(experimentdata) == len(experimentdata.output_data)


@pytest.mark.parametrize("slice_type", [3, [0, 1, 3], slice(0, 3)])
def test_experiment_data_getitem_(slice_type: int | Iterable[int], experimentdata: ExperimentData):
    input_data = experimentdata.input_data[slice_type]
    output_data = experimentdata.output_data[slice_type]
    jobs = experimentdata.jobs[slice_type]
    constructed_experimentdata = ExperimentData._from_object(input_data, output_data, jobs, experimentdata.domain)
    assert constructed_experimentdata == experimentdata[slice_type]

#                                                                           Constructors
# ======================================================================================


def test_from_numpy_array(numpy_array: np.ndarray, experimentdata_continuous: ExperimentData):
    experimentdata_from_numpy = ExperimentData.from_numpy(experimentdata_continuous.domain, numpy_array)
    assert experimentdata_from_numpy == experimentdata_continuous


def test_from_pandas_dataframe(pandas_dataframe: pd.DataFrame, experimentdata_continuous: ExperimentData):
    experimentdata_from_pandas = ExperimentData.from_dataframe(
        pandas_dataframe, domain=experimentdata_continuous.domain)
    assert experimentdata_from_pandas == experimentdata_continuous


def test_from_sampling(experimentdata_continuous: ExperimentData, seed: int):
    sampler = RandomUniform(domain=experimentdata_continuous.domain, number_of_samples=10, seed=seed)
    experimentdata_from_sampling = ExperimentData.from_sampling(sampler)
    assert experimentdata_from_sampling == experimentdata_continuous


@pytest.fixture
def sample_csv_inputdata(tmp_path):
    # Create sample CSV files for testing
    input_csv_file = tmp_path / 'experimentdata_data.csv'

    # Create sample input and output dataframes
    input_data = pd.DataFrame({'input_col1': [1, 2, 3], 'input_col2': [4, 5, 6]})

    return input_csv_file, input_data


@pytest.fixture
def sample_csv_outputdata(tmp_path):
    # Create sample CSV files for testing
    output_csv_file = tmp_path / 'experimentdata_output.csv'

    # Create sample input and output dataframes
    output_data = pd.DataFrame({'output_col1': [7, 8, 9], 'output_col2': [10, 11, 12]})

    return output_csv_file, output_data


def test_from_csv(monkeypatch, sample_csv_inputdata, sample_csv_outputdata):
    input_path, input_data = sample_csv_inputdata
    output_path, output_data = sample_csv_inputdata
    # Define a custom function to replace pd.read_csv

    def mock_read_csv(*args, **kwargs):
        if args[0] == input_path:
            return input_data
        elif args[0] == output_path:
            return output_data
        else:
            raise ValueError(f"Unexpected file path: {args[0]}")

    # Monkeypatch pd.read_csv with the custom function
    monkeypatch.setattr(pd, 'read_csv', mock_read_csv)

    # Call ExperimentData.from_csv
    experiment_data = ExperimentData.from_csv(input_path, output_path)

    # Add assertions to check if experiment_data is created correctly
    assert isinstance(experiment_data, ExperimentData)


def test_from_object(experimentdata_continuous: ExperimentData):
    input_data = experimentdata_continuous.input_data
    output_data = experimentdata_continuous.output_data
    jobs = experimentdata_continuous.jobs
    domain = experimentdata_continuous.domain
    experiment_data = ExperimentData._from_object(input_data, output_data, jobs, domain)
    assert experiment_data == experimentdata_continuous

#                                                                              Exporters
# ======================================================================================


def test_to_numpy(experimentdata_continuous: ExperimentData, numpy_array: np.ndarray):
    x, y = experimentdata_continuous.to_numpy()
    # assert if x and numpy_array have all the same values
    assert np.allclose(x, numpy_array)


def test_to_xarray(experimentdata_continuous: ExperimentData, xarray_dataset: xr.DataSet):
    exported_dataset = experimentdata_continuous.to_xarray()
    # assert if xr_dataset is equal to xarray
    assert exported_dataset.equals(xarray_dataset)

#                                                                              Exporters
# ======================================================================================


def test_add_new_input_column(experimentdata: ExperimentData, continuous_parameter: ContinuousParameter):
    experimentdata.add_new_input_column(name='test', parameter=continuous_parameter)
    assert 'test' in experimentdata.input_data.names


def test_add_new_output_column(experimentdata: ExperimentData):
    experimentdata.add_new_output_column(name='test')
    assert 'test' in experimentdata.output_data.names


def test_add_numpy_arrays(numpy_array: np.ndarray, experimentdata_continuous: ExperimentData):
    exp_data = ExperimentData(experimentdata_continuous.domain)
    exp_data.add_numpy_arrays(numpy_array)
    assert exp_data == experimentdata_continuous


def test_fill_outputs(experimentdata_continuous: ExperimentData,
                      numpy_output_array: np.ndarray, numpy_array: np.ndarray):
    exp_data = ExperimentData(experimentdata_continuous.domain)
    exp_data.add_new_output_column(name='y')
    exp_data.add_numpy_arrays(numpy_array, numpy_output_array)
    experimentdata_continuous.fill_output(numpy_output_array)

    assert exp_data == experimentdata_continuous


def test_set_error(experimentdata_continuous: ExperimentData):
    experimentdata_continuous.set_error(3)
    assert experimentdata_continuous.jobs.jobs[3] == Status.ERROR


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
