from __future__ import annotations

import csv
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from f3dasm import ExperimentData, ExperimentSample
from f3dasm._src.design.parameter import _ContinuousParameter
from f3dasm._src.experimentdata._data import DataTypes, _Data
from f3dasm._src.experimentdata._jobqueue import _JobQueue
from f3dasm.design import Domain, Status, make_nd_continuous_domain

pytestmark = pytest.mark.smoke

SEED = 42


def test_check_experimentdata(experimentdata: ExperimentData):
    assert isinstance(experimentdata, ExperimentData)

# Write test functions


def test_experiment_data_init(experimentdata: ExperimentData, domain: Domain):
    assert experimentdata.domain == domain
    assert experimentdata.project_dir == Path.cwd()
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
    assert len(experimentdata) == len(experimentdata._input_data)


@pytest.mark.parametrize("slice_type", [3, [0, 1, 3]])
def test_experiment_data_select(slice_type: int | Iterable[int], experimentdata: ExperimentData):
    input_data = experimentdata._input_data[slice_type]
    output_data = experimentdata._output_data[slice_type]
    jobs = experimentdata._jobs[slice_type]
    constructed_experimentdata = ExperimentData(
        input_data=input_data, output_data=output_data, jobs=jobs, domain=experimentdata.domain)
    assert constructed_experimentdata == experimentdata.select(slice_type)

#                                                                           Constructors
# ======================================================================================


def test_from_file(experimentdata_continuous: ExperimentData, seed: int, tmp_path: Path):
    # experimentdata_continuous.filename = tmp_path / 'test001'
    experimentdata_continuous.store(tmp_path / 'experimentdata')

    experimentdata_from_file = ExperimentData.from_file(
        tmp_path / 'experimentdata')

    # Check if the input_data attribute of ExperimentData matches the expected_data
    pd.testing.assert_frame_equal(
        experimentdata_continuous._input_data.to_dataframe(), experimentdata_from_file._input_data.to_dataframe(), check_dtype=False, atol=1e-6)
    pd.testing.assert_frame_equal(experimentdata_continuous._output_data.to_dataframe(),
                                  experimentdata_from_file._output_data.to_dataframe())
    pd.testing.assert_series_equal(
        experimentdata_continuous._jobs.jobs, experimentdata_from_file._jobs.jobs)
    # assert experimentdata_continuous.input_data == experimentdata_from_file.input_data
    assert experimentdata_continuous._output_data == experimentdata_from_file._output_data
    assert experimentdata_continuous.domain == experimentdata_from_file.domain
    assert experimentdata_continuous._jobs == experimentdata_from_file._jobs


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
    input_data = pd.DataFrame(
        {'input_col1': [1, 2, 3], 'input_col2': [4, 5, 6]})

    return input_csv_file, input_data


@pytest.fixture
def sample_csv_outputdata(tmp_path):
    # Create sample CSV files for testing
    output_csv_file = tmp_path / 'experimentdata_output.csv'

    # Create sample input and output dataframes
    output_data = pd.DataFrame(
        {'output_col1': [7, 8, 9], 'output_col2': [10, 11, 12]})

    return output_csv_file, output_data


def test_from_object(experimentdata_continuous: ExperimentData):
    input_data = experimentdata_continuous._input_data
    output_data = experimentdata_continuous._output_data
    jobs = experimentdata_continuous._jobs
    domain = experimentdata_continuous.domain
    experiment_data = ExperimentData(
        input_data=input_data, output_data=output_data, jobs=jobs, domain=domain)
    assert experiment_data == ExperimentData(
        input_data=input_data, output_data=output_data, jobs=jobs, domain=domain)
    assert experiment_data == experimentdata_continuous

#                                                                              Exporters
# ======================================================================================


def test_to_numpy(experimentdata_continuous: ExperimentData, numpy_array: np.ndarray):
    x, y = experimentdata_continuous.to_numpy()

    # cast x to floats
    x = x.astype(float)
    # assert if x and numpy_array have all the same values
    assert np.allclose(x, numpy_array)


def test_to_xarray(experimentdata_continuous: ExperimentData, xarray_dataset: xr.DataSet):
    exported_dataset = experimentdata_continuous.to_xarray()
    # assert if xr_dataset is equal to xarray
    assert exported_dataset.equals(xarray_dataset)


def test_to_pandas(experimentdata_continuous: ExperimentData, pandas_dataframe: pd.DataFrame):
    exported_dataframe, _ = experimentdata_continuous.to_pandas()
    # assert if pandas_dataframe is equal to exported_dataframe
    pd.testing.assert_frame_equal(
        exported_dataframe, pandas_dataframe, atol=1e-6, check_dtype=False)
#                                                                              Exporters
# ======================================================================================


def test_add_new_input_column(experimentdata: ExperimentData,
                              continuous_parameter: _ContinuousParameter):
    kwargs = {'low': continuous_parameter.lower_bound,
              'high': continuous_parameter.upper_bound}
    experimentdata.add_input_parameter(
        name='test', type='float', **kwargs)
    assert 'test' in experimentdata._input_data.names


def test_add_new_output_column(experimentdata: ExperimentData):
    experimentdata.add_output_parameter(name='test', is_disk=False)
    assert 'test' in experimentdata._output_data.names


def test_set_error(experimentdata_continuous: ExperimentData):
    experimentdata_continuous._set_error(3)
    assert experimentdata_continuous._jobs.jobs[3] == Status.ERROR


# Helper function to create a temporary CSV file with sample data
def create_sample_csv_input(file_path):
    data = [
        ["x0", "x1", "x2"],
        [0.77395605, 0.43887844, 0.85859792],
        [0.69736803, 0.09417735, 0.97562235],
        [0.7611397, 0.78606431, 0.12811363],
        [0.45038594, 0.37079802, 0.92676499],
        [0.64386512, 0.82276161, 0.4434142],
        [0.22723872, 0.55458479, 0.06381726],
        [0.82763117, 0.6316644, 0.75808774],
        [0.35452597, 0.97069802, 0.89312112],
        [0.7783835, 0.19463871, 0.466721],
        [0.04380377, 0.15428949, 0.68304895],
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

    _data_input = _Data.from_dataframe(pd_input())
    _data_output = _Data.from_dataframe(pd_output())
    experimentdata = ExperimentData(
        domain=domain, input_data=_data_input, output_data=_data_output)
    experimentdata._jobs.store(filepath)


def create_jobs_pickle_open(filepath):
    domain = make_nd_continuous_domain(bounds=np.array([[0., 1.], [0., 1.], [0., 1.]]),
                                       dimensionality=3)

    _data_input = _Data.from_dataframe(pd_input())
    experimentdata = ExperimentData(domain=domain, input_data=_data_input)
    experimentdata._jobs.store(filepath)


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
        [0.77395605, 0.43887844, 0.85859792],
        [0.69736803, 0.09417735, 0.97562235],
        [0.7611397, 0.78606431, 0.12811363],
        [0.45038594, 0.37079802, 0.92676499],
        [0.64386512, 0.82276161, 0.4434142],
        [0.22723872, 0.55458479, 0.06381726],
        [0.82763117, 0.6316644, 0.75808774],
        [0.35452597, 0.97069802, 0.89312112],
        [0.7783835, 0.19463871, 0.466721],
        [0.04380377, 0.15428949, 0.68304895],
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
        [0.77395605, 0.43887844, 0.85859792],
        [0.69736803, 0.09417735, 0.97562235],
        [0.7611397, 0.78606431, 0.12811363],
        [0.45038594, 0.37079802, 0.92676499],
        [0.64386512, 0.82276161, 0.4434142],
        [0.22723872, 0.55458479, 0.06381726],
        [0.82763117, 0.6316644, 0.75808774],
        [0.35452597, 0.97069802, 0.89312112],
        [0.7783835, 0.19463871, 0.466721],
        [0.04380377, 0.15428949, 0.68304895],
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
    return _Data.from_dataframe(pd_input())


def data_output():
    return _Data.from_dataframe(pd_output())


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
            ExperimentData(domain=domain, input_data=input_data,
                           output_data=output_data, jobs=jobs)
        return
    # Initialize ExperimentData with the CSV file
    experiment_data = ExperimentData(domain=domain, input_data=input_data,
                                     output_data=output_data, jobs=jobs)

    # Check if the input_data attribute of ExperimentData matches the expected_data
    pd.testing.assert_frame_equal(
        experiment_data._input_data.to_dataframe(), experimentdata_expected._input_data.to_dataframe(), check_dtype=False, atol=1e-6)
    pd.testing.assert_frame_equal(experiment_data._output_data.to_dataframe(),
                                  experimentdata_expected._output_data.to_dataframe(), check_dtype=False)


@pytest.mark.parametrize("input_data", [pd_input(), path_input, str_input, data_input(), numpy_input()])
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
            ExperimentData(domain=domain, input_data=input_data,
                           output_data=output_data, jobs=jobs)
        return

    # Initialize ExperimentData with the CSV file
    experiment_data = ExperimentData(domain=domain, input_data=input_data,
                                     output_data=output_data, jobs=jobs)

    # Check if the input_data attribute of ExperimentData matches the expected_data
    pd.testing.assert_frame_equal(
        experiment_data._input_data.to_dataframe(), experimentdata_expected_no_output._input_data.to_dataframe(), atol=1e-6, check_dtype=False)
    pd.testing.assert_frame_equal(experiment_data._output_data.to_dataframe(),
                                  experimentdata_expected_no_output._output_data.to_dataframe())
    pd.testing.assert_series_equal(
        experiment_data._jobs.jobs, experimentdata_expected_no_output._jobs.jobs)
    # assert experiment_data.domain == experimentdata_expected_no_output.domain
    assert experiment_data._jobs == experimentdata_expected_no_output._jobs


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
    pd.testing.assert_frame_equal(
        experiment_data._input_data.to_dataframe(), experimentdata_expected_only_domain._input_data.to_dataframe(), check_dtype=False)
    pd.testing.assert_frame_equal(experiment_data._output_data.to_dataframe(),
                                  experimentdata_expected_only_domain._output_data.to_dataframe(), check_dtype=False)
    assert experiment_data._input_data == experimentdata_expected_only_domain._input_data
    assert experiment_data._output_data == experimentdata_expected_only_domain._output_data
    assert experiment_data.domain == experimentdata_expected_only_domain.domain
    assert experiment_data._jobs == experimentdata_expected_only_domain._jobs

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
    assert (tmp_path / "test" / "experiment_data" / "input.csv").exists()
    assert (tmp_path / "test" / "experiment_data" / "output.csv").exists()
    assert (tmp_path / "test" / "experiment_data" / "domain.pkl").exists()
    assert (tmp_path / "test" / "experiment_data" / "jobs.pkl").exists()


def test_store_give_no_filename(experimentdata: ExperimentData, tmp_path: Path):
    experimentdata.set_project_dir(tmp_path / 'test2')
    experimentdata.store()
    assert (tmp_path / "test2" / "experiment_data" / "input.csv").exists()
    assert (tmp_path / "test2" / "experiment_data" / "output.csv").exists()
    assert (tmp_path / "test2" / "experiment_data" / "domain.pkl").exists()
    assert (tmp_path / "test2" / "experiment_data" / "jobs.pkl").exists()


@pytest.mark.parametrize("mode", ["sequential", "parallel", "typo"])
def test_evaluate_mode(mode: str, experimentdata_continuous: ExperimentData, tmp_path: Path):
    experimentdata_continuous.filename = tmp_path / 'test009'

    if mode == "typo":
        with pytest.raises(ValueError):
            experimentdata_continuous.evaluate(
                data_generator="ackley", mode=mode,
                scale_bounds=np.array([[0., 1.], [0., 1.], [0., 1.]]),
                seed=SEED)
    else:
        experimentdata_continuous.evaluate(
            data_generator="ackley", mode=mode,
            scale_bounds=np.array([[0., 1.], [0., 1.], [0., 1.]]),
            seed=SEED)


def test_get_input_data(experimentdata_expected_no_output: ExperimentData):
    input_data = experimentdata_expected_no_output.get_input_data()
    df, _ = input_data.to_pandas()
    pd.testing.assert_frame_equal(df, pd_input(), check_dtype=False, atol=1e-6)
    assert experimentdata_expected_no_output._input_data == input_data._input_data


@pytest.mark.parametrize("selection", ["x0", ["x0"], ["x0", "x2"]])
def test_get_input_data_selection(experimentdata_expected_no_output: ExperimentData, selection: Iterable[str] | str):
    input_data = experimentdata_expected_no_output.get_input_data(selection)
    df, _ = input_data.to_pandas()
    if isinstance(selection, str):
        selection = [selection]
    selected_pd = pd_input()[selection]
    pd.testing.assert_frame_equal(
        df, selected_pd, check_dtype=False, atol=1e-6)


def test_get_output_data(experimentdata_expected: ExperimentData):
    output_data = experimentdata_expected.get_output_data()
    _, df = output_data.to_pandas()
    pd.testing.assert_frame_equal(df, pd_output(), check_dtype=False)
    assert experimentdata_expected._output_data == output_data._output_data


@pytest.mark.parametrize("selection", ["y", ["y"]])
def test_get_output_data_selection(experimentdata_expected: ExperimentData, selection: Iterable[str] | str):
    output_data = experimentdata_expected.get_output_data(selection)
    _, df = output_data.to_pandas()
    if isinstance(selection, str):
        selection = [selection]
    selected_pd = pd_output()[selection]
    pd.testing.assert_frame_equal(df, selected_pd, check_dtype=False)


def test_iter_behaviour(experimentdata_continuous: ExperimentData):
    for i in experimentdata_continuous:
        assert isinstance(i, ExperimentSample)

    selected_experimentdata = experimentdata_continuous.select([0, 2, 4])
    for i in selected_experimentdata:
        assert isinstance(i, ExperimentSample)


def test_select_with_status_open(experimentdata: ExperimentData):
    selected_data = experimentdata.select_with_status('open')
    assert all(job == Status.OPEN for job in selected_data._jobs.jobs)


def test_select_with_status_in_progress(experimentdata: ExperimentData):
    selected_data = experimentdata.select_with_status('in progress')
    assert all(job == Status.IN_PROGRESS for job in selected_data._jobs.jobs)


def test_select_with_status_finished(experimentdata: ExperimentData):
    selected_data = experimentdata.select_with_status('finished')
    assert all(job == Status.FINISHED for job in selected_data._jobs.jobs)


def test_select_with_status_error(experimentdata: ExperimentData):
    selected_data = experimentdata.select_with_status('error')
    assert all(job == Status.ERROR for job in selected_data._jobs.jobs)


def test_select_with_status_invalid_status(experimentdata: ExperimentData):
    with pytest.raises(ValueError):
        _ = experimentdata.select_with_status('invalid_status')


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
