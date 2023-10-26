from __future__ import annotations

import csv
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from f3dasm import ExperimentData
from f3dasm._src.experimentdata.experimentdata import DataTypes
from f3dasm.design import (ContinuousParameter, Domain, Status, _Data,
                           _JobQueue, make_nd_continuous_domain)

pytestmark = pytest.mark.smoke

SEED = 42


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
    experimentdata_total._add_experiments(experimentdata)
    experimentdata_total._add_experiments(experimentdata2)
    assert experimentdata_total == experimentdata + experimentdata2


def test_experiment_data_len_empty(domain: Domain):
    experiment_data = ExperimentData(domain)
    assert len(experiment_data) == 0  # Update with the expected length


def test_experiment_data_len_equals_input_data(experimentdata: ExperimentData):
    assert len(experimentdata) == len(experimentdata.input_data)


def test_experiment_data_len_equals_output_data(experimentdata: ExperimentData):
    assert len(experimentdata) == len(experimentdata.output_data)


@pytest.mark.parametrize("slice_type", [3, [0, 1, 3], slice(0, 3)])
def test_experiment_data_select(slice_type: int | Iterable[int], experimentdata: ExperimentData):
    input_data = experimentdata.input_data[slice_type]
    output_data = experimentdata.output_data[slice_type]
    jobs = experimentdata.jobs[slice_type]
    constructed_experimentdata = ExperimentData(
        input_data=input_data, output_data=output_data, jobs=jobs, domain=experimentdata.domain)
    assert constructed_experimentdata == experimentdata.select(slice_type)

#                                                                           Constructors
# ======================================================================================


def test_from_file(experimentdata_continuous: ExperimentData, seed: int, tmp_path: Path):
    # experimentdata_continuous.filename = tmp_path / 'test001'
    experimentdata_continuous.store(tmp_path / 'experimentdata')

    experimentdata_from_file = ExperimentData.from_file(tmp_path / 'experimentdata')

    # Check if the input_data attribute of ExperimentData matches the expected_data
    pd.testing.assert_frame_equal(experimentdata_continuous.input_data.data, experimentdata_from_file.input_data.data)
    pd.testing.assert_frame_equal(experimentdata_continuous.output_data.data,
                                  experimentdata_from_file.output_data.data)
    pd.testing.assert_series_equal(experimentdata_continuous.jobs.jobs, experimentdata_from_file.jobs.jobs)
    # assert experimentdata_continuous.input_data == experimentdata_from_file.input_data
    assert experimentdata_continuous.output_data == experimentdata_from_file.output_data
    assert experimentdata_continuous.domain == experimentdata_from_file.domain
    assert experimentdata_continuous.jobs == experimentdata_from_file.jobs


def test_from_file_wrong_name(experimentdata_continuous: ExperimentData, seed: int, tmp_path: Path):
    experimentdata_continuous.filename = tmp_path / 'test001'
    experimentdata_continuous.store()

    with pytest.raises(FileNotFoundError):
        _ = ExperimentData.from_file(tmp_path / 'experimentdata')


def test_from_sampling(experimentdata_continuous: ExperimentData, seed: int):
    # sampler = RandomUniform(domain=experimentdata_continuous.domain, number_of_samples=10, seed=seed)
    experimentdata_from_sampling = ExperimentData.from_sampling(sampler='random',
                                                                domain=experimentdata_continuous.domain,
                                                                n_samples=10, seed=seed)
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


def test_from_object(experimentdata_continuous: ExperimentData):
    input_data = experimentdata_continuous.input_data
    output_data = experimentdata_continuous.output_data
    jobs = experimentdata_continuous.jobs
    domain = experimentdata_continuous.domain
    experiment_data = ExperimentData(input_data=input_data, output_data=output_data, jobs=jobs, domain=domain)
    assert experiment_data == ExperimentData(input_data=input_data, output_data=output_data, jobs=jobs, domain=domain)
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


def test_to_pandas(experimentdata_continuous: ExperimentData, pandas_dataframe: pd.DataFrame):
    exported_dataframe, _ = experimentdata_continuous.to_pandas()
    # assert if pandas_dataframe is equal to exported_dataframe
    assert exported_dataframe.equals(pandas_dataframe)
#                                                                              Exporters
# ======================================================================================


def test_add_new_input_column(experimentdata: ExperimentData, continuous_parameter: ContinuousParameter):
    experimentdata.add_input_parameter(name='test', parameter=continuous_parameter)
    assert 'test' in experimentdata.input_data.names


def test_add_new_output_column(experimentdata: ExperimentData):
    experimentdata.add_output_parameter(name='test')
    assert 'test' in experimentdata.output_data.names


def test_fill_outputs(experimentdata_continuous: ExperimentData,
                      numpy_output_array: np.ndarray, numpy_array: np.ndarray):
    exp_data = ExperimentData(experimentdata_continuous.domain)
    exp_data.add_output_parameter(name='y')
    exp_data.add(domain=exp_data.domain, input_data=numpy_array, output_data=numpy_output_array)
    experimentdata_continuous.fill_output(numpy_output_array)

    assert exp_data == experimentdata_continuous


def test_set_error(experimentdata_continuous: ExperimentData):
    experimentdata_continuous._set_error(3)
    assert experimentdata_continuous.jobs.jobs[3] == Status.ERROR


# Helper function to create a temporary CSV file with sample data
def create_sample_csv_input(file_path):
    data = [
        ["x0", "x1", "x2"],
        [0.374540, 0.950714, 0.731994],
        [0.598658, 0.156019, 0.155995],
        [0.058084, 0.866176, 0.601115],
        [0.708073, 0.020584, 0.969910],
        [0.832443, 0.212339, 0.181825],
        [0.183405, 0.304242, 0.524756],
        [0.431945, 0.291229, 0.611853],
        [0.139494, 0.292145, 0.366362],
        [0.456070, 0.785176, 0.199674],
        [0.514234, 0.592415, 0.046450],
        [0.000000, 0.000000, 0.000000],
        [1.000000, 1.000000, 1.000000],
    ]
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)


def create_sample_csv_output(file_path):
    data = [
        ["y"],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],

    ]
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

# Pytest fixture to create a temporary CSV file


def create_domain_pickle(filepath):
    domain = make_nd_continuous_domain(bounds=np.array([[0., 1.], [0., 1.], [0., 1.]]),
                                       dimensionality=3)
    domain.store(filepath)


def create_jobs_pickle_finished(filepath):
    domain = make_nd_continuous_domain(bounds=np.array([[0., 1.], [0., 1.], [0., 1.]]),
                                       dimensionality=3)

    _data_input = _Data(pd_input())
    _data_output = _Data(pd_output())
    experimentdata = ExperimentData(domain=domain, input_data=_data_input, output_data=_data_output)
    experimentdata.jobs.store(filepath)


def create_jobs_pickle_open(filepath):
    domain = make_nd_continuous_domain(bounds=np.array([[0., 1.], [0., 1.], [0., 1.]]),
                                       dimensionality=3)

    _data_input = _Data(pd_input())
    experimentdata = ExperimentData(domain=domain, input_data=_data_input)
    experimentdata.jobs.store(filepath)


def path_domain(tmp_path):
    domain_file_path = tmp_path / "test_domain.pkl"
    create_domain_pickle(domain_file_path)
    return domain_file_path


def str_domain(tmp_path):
    domain_file_path = tmp_path / "test_domain.pkl"
    create_domain_pickle(domain_file_path)
    return str(domain_file_path)


def path_jobs_finished(tmp_path):
    jobs_file_path = tmp_path / "test_jobs.pkl"
    create_jobs_pickle_finished(jobs_file_path)
    return jobs_file_path


def str_jobs_finished(tmp_path):
    jobs_file_path = tmp_path / "test_jobs.pkl"
    create_jobs_pickle_finished(jobs_file_path)
    return str(jobs_file_path)


def path_jobs_open(tmp_path):
    jobs_file_path = tmp_path / "test_jobs.pkl"
    create_jobs_pickle_open(jobs_file_path)
    return jobs_file_path


def str_jobs_open(tmp_path):
    jobs_file_path = tmp_path / "test_jobs.pkl"
    create_jobs_pickle_open(jobs_file_path)
    return str(jobs_file_path)


def path_input(tmp_path):
    csv_file_path = tmp_path / "test_input.csv"
    create_sample_csv_input(csv_file_path)
    return csv_file_path


def str_input(tmp_path):
    csv_file_path = tmp_path / "test_input.csv"
    create_sample_csv_input(csv_file_path)
    return str(csv_file_path)


def path_output(tmp_path: Path):
    csv_file_path = tmp_path / "test_output.csv"
    create_sample_csv_output(csv_file_path)
    return csv_file_path


def str_output(tmp_path: Path):
    csv_file_path = tmp_path / "test_output.csv"
    create_sample_csv_output(csv_file_path)
    return str(csv_file_path)

# Pytest test function for reading and monkeypatching a CSV file


def numpy_input():
    return np.array([
        [0.374540, 0.950714, 0.731994],
        [0.598658, 0.156019, 0.155995],
        [0.058084, 0.866176, 0.601115],
        [0.708073, 0.020584, 0.969910],
        [0.832443, 0.212339, 0.181825],
        [0.183405, 0.304242, 0.524756],
        [0.431945, 0.291229, 0.611853],
        [0.139494, 0.292145, 0.366362],
        [0.456070, 0.785176, 0.199674],
        [0.514234, 0.592415, 0.046450],
        [0.000000, 0.000000, 0.000000],
        [1.000000, 1.000000, 1.000000],
    ])


def numpy_output():
    return np.array([
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],

    ])


def pd_input():
    return pd.DataFrame([
        [0.374540, 0.950714, 0.731994],
        [0.598658, 0.156019, 0.155995],
        [0.058084, 0.866176, 0.601115],
        [0.708073, 0.020584, 0.969910],
        [0.832443, 0.212339, 0.181825],
        [0.183405, 0.304242, 0.524756],
        [0.431945, 0.291229, 0.611853],
        [0.139494, 0.292145, 0.366362],
        [0.456070, 0.785176, 0.199674],
        [0.514234, 0.592415, 0.046450],
        [0.000000, 0.000000, 0.000000],
        [1.000000, 1.000000, 1.000000],
    ], columns=["x0", "x1", "x2"])


def pd_output():
    return pd.DataFrame([
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],
        [0.0],

    ], columns=["y"])


def data_input():
    return _Data(pd_input())


def data_output():
    return _Data(pd_output())


@pytest.mark.parametrize("input_data", [path_input, str_input, pd_input(), data_input(), numpy_input()])
@pytest.mark.parametrize("output_data", [path_output, str_output, pd_output(), data_output()])
@pytest.mark.parametrize("domain", [make_nd_continuous_domain(bounds=np.array([[0., 1.], [0., 1.], [0., 1.]]),
                                                              dimensionality=3), None, path_domain, str_domain])
@pytest.mark.parametrize("jobs", [None, path_jobs_finished, str_jobs_finished])
def test_init_with_output(input_data: DataTypes, output_data: DataTypes, domain: Domain | str | Path | None,
                          jobs: _JobQueue | str | Path | None,
                          experimentdata_expected: ExperimentData, monkeypatch, tmp_path: Path):

    # if input_data is Callable
    if callable(input_data):
        input_data = input_data(tmp_path)
        expected_data_input = pd.read_csv(input_data)

    # if output_data is Callable
    if callable(output_data):
        output_data = output_data(tmp_path)
        expected_data_output = pd.read_csv(output_data)

    if callable(domain):
        domain = domain(tmp_path)
        expected_domain = Domain.from_file(domain)

    if callable(jobs):
        jobs = jobs(tmp_path)
        expected_jobs = _JobQueue.from_file(jobs).jobs

    # monkeypatch pd.read_csv to return the expected_data DataFrame
    def mock_read_csv(*args, **kwargs):

        path = args[0]
        if isinstance(args[0], str):
            path = Path(path)

        if path == tmp_path / "test_input.csv":
            return expected_data_input

        elif path == tmp_path / "test_output.csv":
            return expected_data_output

        else:
            raise ValueError("Unexpected file path")

    def mock_load_pickle(*args, **kwargs):
        return expected_domain

    def mock_pd_read_pickle(*args, **kwargs):
        path = args[0]

        if isinstance(path, str):
            path = Path(path)

        if path == tmp_path / "test_jobs.pkl":
            return expected_jobs

        else:
            raise ValueError("Unexpected jobs file path")

    monkeypatch.setattr(pd, "read_csv", mock_read_csv)
    monkeypatch.setattr(pickle, "load", mock_load_pickle)
    monkeypatch.setattr(pd, "read_pickle", mock_pd_read_pickle)

    if isinstance(input_data, np.ndarray) and domain is None:
        with pytest.raises(ValueError):
            ExperimentData(domain=domain, input_data=input_data, output_data=output_data, jobs=jobs)
        return
    # Initialize ExperimentData with the CSV file
    experiment_data = ExperimentData(domain=domain, input_data=input_data,
                                     output_data=output_data, jobs=jobs)

    # Check if the input_data attribute of ExperimentData matches the expected_data
    pd.testing.assert_frame_equal(experiment_data.input_data.data, experimentdata_expected.input_data.data)
    pd.testing.assert_frame_equal(experiment_data.output_data.data,
                                  experimentdata_expected.output_data.data)
    assert experiment_data == experimentdata_expected


@pytest.mark.parametrize("input_data", [path_input, str_input, pd_input(), data_input(), numpy_input()])
@pytest.mark.parametrize("output_data", [None])
@pytest.mark.parametrize("domain", [make_nd_continuous_domain(bounds=np.array([[0., 1.], [0., 1.], [0., 1.]]),
                                                              dimensionality=3), None, path_domain, str_domain])
@pytest.mark.parametrize("jobs", [None, path_jobs_open, str_jobs_open])
def test_init_without_output(input_data: DataTypes, output_data: DataTypes, domain: Domain, jobs: _JobQueue,
                             experimentdata_expected_no_output: ExperimentData, monkeypatch, tmp_path):

    # if input_data is Callable
    if callable(input_data):
        input_data = input_data(tmp_path)
        expected_data_input = pd.read_csv(input_data)

    # if output_data is Callable
    if callable(output_data):
        output_data = output_data(tmp_path)
        expected_data_output = pd.read_csv(output_data)

    if callable(domain):
        domain = domain(tmp_path)
        expected_domain = Domain.from_file(domain)

    if callable(jobs):
        jobs = jobs(tmp_path)
        expected_jobs = _JobQueue.from_file(jobs).jobs

    # monkeypatch pd.read_csv to return the expected_data DataFrame
    def mock_read_csv(*args, **kwargs):

        path = args[0]
        if isinstance(args[0], str):
            path = Path(path)

        if path == tmp_path / "test_input.csv":
            return expected_data_input

        elif path == tmp_path / "test_output.csv":
            return expected_data_output

        else:
            raise ValueError("Unexpected file path")

    def mock_load_pickle(*args, **kwargs):
        return expected_domain

    def mock_pd_read_pickle(*args, **kwargs):
        path = args[0]

        if isinstance(path, str):
            path = Path(path)

        if path == tmp_path / "test_jobs.pkl":
            return expected_jobs

    monkeypatch.setattr(pd, "read_csv", mock_read_csv)
    monkeypatch.setattr(pickle, "load", mock_load_pickle)
    monkeypatch.setattr(pd, "read_pickle", mock_pd_read_pickle)

    if isinstance(input_data, np.ndarray) and domain is None:
        with pytest.raises(ValueError):
            ExperimentData(domain=domain, input_data=input_data, output_data=output_data, jobs=jobs)
        return

    # Initialize ExperimentData with the CSV file
    experiment_data = ExperimentData(domain=domain, input_data=input_data,
                                     output_data=output_data, jobs=jobs)

    # Check if the input_data attribute of ExperimentData matches the expected_data
    pd.testing.assert_frame_equal(experiment_data.input_data.data, experimentdata_expected_no_output.input_data.data)
    pd.testing.assert_frame_equal(experiment_data.output_data.data,
                                  experimentdata_expected_no_output.output_data.data)
    pd.testing.assert_series_equal(experiment_data.jobs.jobs, experimentdata_expected_no_output.jobs.jobs)
    assert experiment_data.input_data == experimentdata_expected_no_output.input_data
    assert experiment_data.output_data == experimentdata_expected_no_output.output_data
    assert experiment_data.domain == experimentdata_expected_no_output.domain
    assert experiment_data.jobs == experimentdata_expected_no_output.jobs

    assert experiment_data == experimentdata_expected_no_output


@pytest.mark.parametrize("input_data", [None])
@pytest.mark.parametrize("output_data", [None])
@pytest.mark.parametrize("domain", [make_nd_continuous_domain(bounds=np.array([[0., 1.], [0., 1.], [0., 1.]]),
                                                              dimensionality=3), path_domain, str_domain])
def test_init_only_domain(input_data: DataTypes, output_data: DataTypes, domain: Domain | str | Path,
                          experimentdata_expected_only_domain: ExperimentData,
                          monkeypatch, tmp_path):

    # if input_data is Callable
    if callable(input_data):
        input_data = input_data(tmp_path)
        expected_data_input = pd.read_csv(input_data)

    # if output_data is Callable
    if callable(output_data):
        output_data = output_data(tmp_path)
        expected_data_output = pd.read_csv(output_data)

    if callable(domain):
        domain = domain(tmp_path)
        expected_domain = Domain.from_file(domain)

    # monkeypatch pd.read_csv to return the expected_data DataFrame
    def mock_read_csv(*args, **kwargs):

        path = args[0]
        if isinstance(args[0], str):
            path = Path(path)

        if path == tmp_path / "test_input.csv":
            return expected_data_input

        elif path == tmp_path / "test_output.csv":
            return expected_data_output

        else:
            raise ValueError("Unexpected file path")

    def mock_load_pickle(*args, **kwargs):
        return expected_domain

    monkeypatch.setattr(pd, "read_csv", mock_read_csv)
    monkeypatch.setattr(pickle, "load", mock_load_pickle)

    # Initialize ExperimentData with the CSV file
    experiment_data = ExperimentData(domain=domain, input_data=input_data,
                                     output_data=output_data)

    # Check if the input_data attribute of ExperimentData matches the expected_data
    pd.testing.assert_frame_equal(experiment_data.input_data.data, experimentdata_expected_only_domain.input_data.data)
    pd.testing.assert_frame_equal(experiment_data.output_data.data,
                                  experimentdata_expected_only_domain.output_data.data)
    assert experiment_data.input_data == experimentdata_expected_only_domain.input_data
    assert experiment_data.output_data == experimentdata_expected_only_domain.output_data
    assert experiment_data.domain == experimentdata_expected_only_domain.domain
    assert experiment_data.jobs == experimentdata_expected_only_domain.jobs

    assert experiment_data == experimentdata_expected_only_domain


@pytest.mark.parametrize("input_data", [[0.1, 0.2], {"a": 0.1, "b": 0.2}, 0.2, 2])
def test_invalid_type(input_data):
    with pytest.raises(TypeError):
        ExperimentData(input_data=input_data)


def test_add_invalid_type(experimentdata: ExperimentData):
    with pytest.raises(TypeError):
        experimentdata + 1


def test_add_two_different_domains(experimentdata: ExperimentData, experimentdata_continuous: ExperimentData):
    with pytest.raises(ValueError):
        experimentdata + experimentdata_continuous


def test_repr_html(experimentdata: ExperimentData, monkeypatch):
    assert isinstance(experimentdata._repr_html_(), str)


def test_store(experimentdata: ExperimentData, tmp_path: Path):
    experimentdata.store(tmp_path / "test")
    assert (tmp_path / "test_input.csv").exists()
    assert (tmp_path / "test_output.csv").exists()
    assert (tmp_path / "test_domain.pkl").exists()
    assert (tmp_path / "test_jobs.pkl").exists()


def test_store_give_no_filename(experimentdata: ExperimentData, tmp_path: Path):
    experimentdata.filename = tmp_path / 'test2'
    experimentdata.store()
    assert (tmp_path / "test2_input.csv").exists()
    assert (tmp_path / "test2_output.csv").exists()
    assert (tmp_path / "test2_domain.pkl").exists()
    assert (tmp_path / "test2_jobs.pkl").exists()


@pytest.mark.parametrize("mode", ["sequential", "parallel", "typo"])
def test_evaluate_mode(mode: str, experimentdata_continuous: ExperimentData, tmp_path: Path):
    experimentdata_continuous.filename = tmp_path / 'test009'

    if mode == "typo":
        with pytest.raises(ValueError):
            experimentdata_continuous.evaluate("ackley", mode=mode, kwargs={
                                               "scale_bounds": np.array([[0., 1.], [0., 1.], [0., 1.]]), 'seed': SEED})
    else:
        experimentdata_continuous.evaluate("ackley", mode=mode, kwargs={
            "scale_bounds": np.array([[0., 1.], [0., 1.], [0., 1.]]), 'seed': SEED})


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
