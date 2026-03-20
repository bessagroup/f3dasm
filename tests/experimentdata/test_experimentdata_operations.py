"""Tests for ExperimentData operations: manipulation, jobs, conversion."""

from copy import copy, deepcopy

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from f3dasm import ExperimentData, datagenerator
from f3dasm._src.experimentsample import ExperimentSample, JobStatus
from f3dasm.design import Domain, make_nd_continuous_domain

pytestmark = pytest.mark.smoke

SEED = 42


@pytest.fixture
def domain_3d():
    return make_nd_continuous_domain(
        bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
    )


@pytest.fixture
def experiment_data(domain_3d):
    from f3dasm import create_sampler

    data = ExperimentData(domain=domain_3d)
    sampler = create_sampler(sampler="random", seed=SEED)
    sampler.arm(data=data)
    data = sampler.call(data=data, n_samples=5)
    return data


@pytest.fixture
def experiment_data_with_output(experiment_data):
    @datagenerator(output_names="y")
    def f(x0, x1, x2):
        return x0 + x1 + x2

    return f.call(data=experiment_data)


# ======================= Length and indexing =======================


def test_len(experiment_data):
    assert len(experiment_data) == 5


def test_len_empty():
    data = ExperimentData()
    assert len(data) == 0


def test_getitem_single(experiment_data):
    subset = experiment_data[0]
    assert len(subset) == 1


def test_getitem_multiple(experiment_data):
    subset = experiment_data[[0, 1, 2]]
    assert len(subset) == 3


def test_select(experiment_data):
    subset = experiment_data.select([0, 1])
    assert len(subset) == 2


# ======================= Iteration =======================


def test_iter(experiment_data):
    ids = []
    for id, sample in experiment_data:
        ids.append(id)
        assert isinstance(sample, ExperimentSample)
    assert len(ids) == 5


# ======================= Addition =======================


def test_add_two_experiment_data(experiment_data, domain_3d):
    from f3dasm import create_sampler

    data2 = ExperimentData(domain=domain_3d)
    sampler = create_sampler(sampler="random", seed=99)
    sampler.arm(data=data2)
    data2 = sampler.call(data=data2, n_samples=3)

    combined = experiment_data + data2
    assert len(combined) == 8


def test_add_experiment_sample(experiment_data):
    sample = ExperimentSample(
        _input_data={"x0": 0.5, "x1": 0.5, "x2": 0.5}
    )
    result = experiment_data.add_experiments(sample)
    assert len(result) == 6


def test_add_experiments_invalid_type(experiment_data):
    with pytest.raises(ValueError):
        experiment_data.add_experiments("invalid")


def test_add_experiments_in_place(experiment_data):
    sample = ExperimentSample(
        _input_data={"x0": 0.5, "x1": 0.5, "x2": 0.5}
    )
    result = experiment_data.add_experiments(sample, in_place=True)
    assert result is None
    assert len(experiment_data) == 6


# ======================= Remove rows =======================


def test_remove_rows_bottom(experiment_data):
    result = experiment_data.remove_rows_bottom(2)
    assert len(result) == 3
    assert len(experiment_data) == 5  # original unchanged


def test_remove_rows_bottom_in_place(experiment_data):
    experiment_data.remove_rows_bottom(2, in_place=True)
    assert len(experiment_data) == 3


# ======================= Reset index =======================


def test_reset_index(experiment_data):
    # Remove some rows, then reset
    trimmed = experiment_data.remove_rows_bottom(2)
    reset = trimmed.reset_index()
    assert list(reset.index) == [0, 1, 2]


# ======================= Join =======================


def test_join_two_experiment_data():
    domain1 = Domain()
    domain1.add_float("x0", 0.0, 1.0)
    data1 = ExperimentData(
        domain=domain1, input_data=[{"x0": 0.1}, {"x0": 0.2}]
    )

    domain2 = Domain()
    domain2.add_float("x1", 0.0, 1.0)
    data2 = ExperimentData(
        domain=domain2, input_data=[{"x1": 0.3}, {"x1": 0.4}]
    )

    joined = data1.join(data2)
    assert len(joined) == 2
    # Check both columns present
    df_input, _ = joined.to_pandas()
    assert "x0" in df_input.columns
    assert "x1" in df_input.columns


def test_join_empty_with_nonempty():
    domain = Domain()
    domain.add_float("x0", 0.0, 1.0)

    empty = ExperimentData(domain=domain)
    nonempty = ExperimentData(
        domain=domain, input_data=[{"x0": 0.5}]
    )

    result = empty.join(nonempty)
    assert len(result) == 1


# ======================= Sort =======================


def test_sort(experiment_data_with_output):
    sorted_data = experiment_data_with_output.sort(
        criterion=lambda es: es.output_data.get("y", float("inf"))
    )
    # Verify sorting order
    _, df_out = sorted_data.to_pandas()
    values = df_out["y"].values
    assert all(values[i] <= values[i + 1] for i in range(len(values) - 1))


def test_sort_reverse(experiment_data_with_output):
    sorted_data = experiment_data_with_output.sort(
        criterion=lambda es: es.output_data.get("y", float("inf")),
        reverse=True,
    )
    _, df_out = sorted_data.to_pandas()
    values = df_out["y"].values
    assert all(values[i] >= values[i + 1] for i in range(len(values) - 1))


# ======================= n best =======================


def test_get_n_best_output(experiment_data_with_output):
    best = experiment_data_with_output.get_n_best_output(
        n_samples=2, output_name="y"
    )
    assert len(best) == 2


# ======================= Job status =======================


def test_mark_single(experiment_data):
    result = experiment_data.mark(0, "finished")
    sample = result.get_experiment_sample(0)
    assert sample.job_status == JobStatus.FINISHED


def test_mark_multiple(experiment_data):
    result = experiment_data.mark([0, 1], "in_progress")
    assert result.get_experiment_sample(0).job_status == JobStatus.IN_PROGRESS
    assert result.get_experiment_sample(1).job_status == JobStatus.IN_PROGRESS


def test_mark_all(experiment_data):
    result = experiment_data.mark_all("finished")
    for _, es in result:
        assert es.job_status == JobStatus.FINISHED


def test_mark_all_in_place(experiment_data):
    experiment_data.mark_all("finished", in_place=True)
    for _, es in experiment_data:
        assert es.job_status == JobStatus.FINISHED


def test_get_open_job(experiment_data):
    job_id, sample, domain = experiment_data.get_open_job()
    assert job_id is not None
    assert sample.job_status == JobStatus.IN_PROGRESS


def test_get_open_job_none_when_all_finished(experiment_data):
    experiment_data.mark_all("finished", in_place=True)
    job_id, sample, domain = experiment_data.get_open_job()
    assert job_id is None


def test_is_all_finished(experiment_data):
    assert not experiment_data.is_all_finished()
    experiment_data.mark_all("finished", in_place=True)
    assert experiment_data.is_all_finished()


def test_select_with_status(experiment_data):
    experiment_data.mark(0, "finished", in_place=True)
    finished = experiment_data.select_with_status("finished")
    assert len(finished) == 1

    open_jobs = experiment_data.select_with_status("open")
    assert len(open_jobs) == 4


# ======================= Format conversions =======================


def test_to_numpy(experiment_data_with_output):
    input_arr, output_arr = experiment_data_with_output.to_numpy()
    assert isinstance(input_arr, np.ndarray)
    assert isinstance(output_arr, np.ndarray)
    assert input_arr.shape[0] == len(experiment_data_with_output)
    assert output_arr.shape[0] == len(experiment_data_with_output)


def test_to_pandas(experiment_data_with_output):
    df_input, df_output = experiment_data_with_output.to_pandas()
    assert isinstance(df_input, pd.DataFrame)
    assert isinstance(df_output, pd.DataFrame)
    assert len(df_input) == len(experiment_data_with_output)


def test_to_xarray(experiment_data_with_output):
    ds = experiment_data_with_output.to_xarray()
    assert isinstance(ds, xr.Dataset)
    assert "input" in ds
    assert "output" in ds


def test_to_multiindex(experiment_data_with_output):
    df = experiment_data_with_output.to_multiindex()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(experiment_data_with_output)


# ======================= Copy =======================


def test_copy(experiment_data):
    copied = copy(experiment_data)
    assert copied == experiment_data


def test_copy_in_place(experiment_data):
    result = experiment_data._copy(in_place=True)
    assert result is experiment_data


# ======================= Replace NaN and round =======================


def test_replace_nan():
    sample = ExperimentSample(
        _input_data={"x0": float("nan")},
        _output_data={"y": float("nan")},
    )
    sample.replace_nan(0.0)
    assert sample.input_data["x0"] == 0.0
    assert sample.output_data["y"] == 0.0


def test_round(experiment_data_with_output):
    rounded = experiment_data_with_output.round(2)
    _, df_out = rounded.to_pandas()
    for val in df_out["y"].values:
        # Check that rounding to 2 decimals works
        assert val == round(val, 2)


# ======================= Properties =======================


def test_index_property(experiment_data):
    idx = experiment_data.index
    assert isinstance(idx, pd.Index)
    assert len(idx) == 5


def test_jobs_property(experiment_data):
    jobs = experiment_data.jobs
    assert isinstance(jobs, pd.Series)
    assert len(jobs) == 5


def test_domain_property(experiment_data, domain_3d):
    assert experiment_data.domain is not None
    assert isinstance(experiment_data.domain, Domain)


def test_project_dir_setter(experiment_data, tmp_path):
    experiment_data.project_dir = tmp_path
    assert experiment_data.project_dir == tmp_path


# ======================= Store and load roundtrip =======================


def test_store_and_load_roundtrip(experiment_data_with_output, tmp_path):
    experiment_data_with_output.store(project_dir=tmp_path)
    loaded = ExperimentData.from_file(project_dir=tmp_path)
    assert len(loaded) == len(experiment_data_with_output)


# ======================= Equality =======================


def test_equality(experiment_data):
    copied = experiment_data._copy(in_place=False, deep=True)
    assert copied == experiment_data


def test_inequality(experiment_data):
    other = ExperimentData()
    assert experiment_data != other


# ======================= Constructor variants =======================


def test_init_from_numpy():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    data = ExperimentData(input_data=arr)
    assert len(data) == 2


def test_init_from_dataframe():
    df = pd.DataFrame({"x0": [1.0, 2.0], "x1": [3.0, 4.0]})
    data = ExperimentData(input_data=df)
    assert len(data) == 2


def test_init_with_domain():
    domain = make_nd_continuous_domain(bounds=[[0.0, 1.0]])
    data = ExperimentData(domain=domain)
    assert len(data) == 0
    assert data.domain is not None


def test_init_from_list_of_dicts():
    data = ExperimentData(
        input_data=[{"x0": 1.0}, {"x0": 2.0}]
    )
    assert len(data) == 2


# ======================= get_experiment_sample =======================


def test_get_experiment_sample(experiment_data):
    sample = experiment_data.get_experiment_sample(0)
    assert isinstance(sample, ExperimentSample)


# ======================= set_project_dir =======================


def test_set_project_dir(experiment_data, tmp_path):
    result = experiment_data.set_project_dir(tmp_path)
    assert result.project_dir == tmp_path
    # Original unchanged
    assert experiment_data.project_dir != tmp_path


def test_set_project_dir_in_place(experiment_data, tmp_path):
    experiment_data.set_project_dir(tmp_path, in_place=True)
    assert experiment_data.project_dir == tmp_path
