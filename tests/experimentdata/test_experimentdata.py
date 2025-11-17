from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig

from f3dasm import ExperimentData, create_sampler, datagenerator
from f3dasm.design import Domain, make_nd_continuous_domain

SEED = 42

pytestmark = pytest.mark.smoke


def domain_dictconfig_with_output():
    config_dict = {
        "input": {
            "x0": {"type": "float", "low": 0.0, "high": 1.0},
            "x1": {"type": "float", "low": 0.0, "high": 1.0},
            "x2": {"type": "float", "low": 0.0, "high": 1.0},
        },
        "output": {"y": {}},
    }

    return DictConfig(config_dict)


def domain_dictconfig_without_output():
    config_dict = {
        "input": {
            "x0": {"type": "float", "low": 0.0, "high": 1.0},
            "x1": {"type": "float", "low": 0.0, "high": 1.0},
            "x2": {"type": "float", "low": 0.0, "high": 1.0},
        },
    }

    return DictConfig(config_dict)


def edata_domain_with_output() -> Domain:
    return experiment_data_with_output()._domain


def edata_domain_without_output() -> Domain:
    return experiment_data_without_output()._domain


# =============================================================================


def test_project_dir_false_data():
    with pytest.raises(TypeError):
        ExperimentData(project_dir=0)


# =============================================================================


def experiment_data_with_output() -> ExperimentData:
    domain = make_nd_continuous_domain(
        bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
    )

    data = ExperimentData(domain=domain)

    sampler = create_sampler(sampler="random", seed=SEED)
    sampler.arm(data=data)

    data = sampler.call(data=data, n_samples=10)

    data += ExperimentData(
        input_data=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        domain=data._domain,
    )

    @datagenerator(output_names="y")
    def f(*args, **kwargs):
        return 0.0

    data = f.call(data=data)
    data = data.round(3)

    data = data.set_project_dir("./test_project")
    return data


def experiment_data_without_output() -> ExperimentData:
    domain = make_nd_continuous_domain(
        bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
    )

    data = ExperimentData(domain=domain)

    sampler = create_sampler(sampler="random", seed=SEED)
    sampler.arm(data=data)

    data = sampler.call(data=data, n_samples=10)

    data += ExperimentData(
        input_data=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        domain=data._domain,
    )

    data = data.round(3)

    data = data.set_project_dir("./test_project")
    return data


# =============================================================================


@pytest.fixture(scope="package")
def edata_expected_with_output() -> ExperimentData:
    domain = make_nd_continuous_domain(
        bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
    )

    data = ExperimentData(domain=domain)

    sampler = create_sampler(sampler="random", seed=SEED)
    sampler.arm(data=data)

    data = sampler.call(data=data, n_samples=10)

    data += ExperimentData(
        input_data=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        domain=data._domain,
    )

    @datagenerator(output_names="y")
    def f(*args, **kwargs):
        return 0.0

    data = f.call(data=data)
    # data.evaluate(data_generator=f, output_names=['y'])
    data = data.round(3)

    data = data.set_project_dir("./test_project")

    return data


@pytest.fixture(scope="package")
def edata_expected_without_output() -> ExperimentData:
    domain = make_nd_continuous_domain(
        bounds=[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]
    )

    data = ExperimentData(domain=domain)

    sampler = create_sampler(sampler="random", seed=SEED)
    sampler.arm(data=data)

    data = sampler.call(data=data, n_samples=10)

    data += ExperimentData(
        input_data=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        domain=data._domain,
    )

    data = data.round(3)

    data = data.set_project_dir("./test_project")

    return data


# =============================================================================


def edata_jobs_with_output() -> pd.Series:
    return experiment_data_with_output().jobs


def edata_jobs_without_output() -> pd.Series:
    return experiment_data_without_output().jobs


# =============================================================================


def arr_input():
    return np.array(
        [
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
    ).round(3)


def list_of_dicts_input():
    return [
        {"x0": 0.77395605, "x1": 0.43887844, "x2": 0.85859792},
        {"x0": 0.69736803, "x1": 0.09417735, "x2": 0.97562235},
        {"x0": 0.7611397, "x1": 0.78606431, "x2": 0.12811363},
        {"x0": 0.45038594, "x1": 0.37079802, "x2": 0.92676499},
        {"x0": 0.64386512, "x1": 0.82276161, "x2": 0.4434142},
        {"x0": 0.22723872, "x1": 0.55458479, "x2": 0.06381726},
        {"x0": 0.82763117, "x1": 0.6316644, "x2": 0.75808774},
        {"x0": 0.35452597, "x1": 0.97069802, "x2": 0.89312112},
        {"x0": 0.7783835, "x1": 0.19463871, "x2": 0.466721},
        {"x0": 0.04380377, "x1": 0.15428949, "x2": 0.68304895},
        {"x0": 0.000000, "x1": 0.000000, "x2": 0.000000},
        {"x0": 1.000000, "x1": 1.000000, "x2": 1.000000},
    ]


def list_of_dicts_output():
    return [
        {"y": 0.0},
        {"y": 0.0},
        {"y": 0.0},
        {"y": 0.0},
        {"y": 0.0},
        {"y": 0.0},
        {"y": 0.0},
        {"y": 0.0},
        {"y": 0.0},
        {"y": 0.0},
        {"y": 0.0},
        {"y": 0.0},
    ]


def pd_input():
    return pd.DataFrame(arr_input(), columns=["x0", "x1", "x2"])


def arr_output():
    return np.array(
        [
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
    )


def pd_output():
    return pd.DataFrame(arr_output(), columns=["y"])


@pytest.mark.parametrize(
    "input_data",
    ["test_input.csv", pd_input(), arr_input(), list_of_dicts_input()],
)
@pytest.mark.parametrize(
    "output_data",
    ["test_output.csv", pd_output(), arr_output(), list_of_dicts_output()],
)
@pytest.mark.parametrize(
    "domain",
    [
        "test_domain.json",
        edata_domain_with_output(),
        None,
        domain_dictconfig_with_output(),
    ],
)
@pytest.mark.parametrize(
    "jobs", [edata_jobs_with_output(), "test_jobs.csv", None]
)
@pytest.mark.parametrize(
    "project_dir", ["./test_project", Path("./test_project")]
)
def test_experimentdata_creation_with_output(
    input_data,
    output_data,
    domain,
    jobs,
    project_dir,
    edata_expected_with_output,
    monkeypatch,
):
    def mock_read_csv(path, *args, **kwargs):
        if str(path) == "test_input.csv":
            return pd_input()
        elif str(path) == "test_output.csv":
            return pd_output()
        elif str(path) == "test_jobs.csv":
            return edata_jobs_with_output()
        raise ValueError(f"Unexpected file path: {path}")

    def mock_domain_from_file(path, *args, **kwargs):
        return edata_domain_with_output()

    monkeypatch.setattr(pd, "read_csv", mock_read_csv)
    monkeypatch.setattr(Domain, "from_file", mock_domain_from_file)
    mock_stat = MagicMock()
    mock_stat.st_size = 10  # Non Empty file

    with (
        patch.object(Path, "exists", return_value=True),
        patch.object(Path, "stat", return_value=mock_stat),
    ):
        experiment_data = ExperimentData(
            domain=domain,
            input_data=input_data,
            output_data=output_data,
            jobs=jobs,
            project_dir=project_dir,
        )

    if domain is None:
        experiment_data._domain = edata_domain_with_output()

    experiment_data = experiment_data.round(3)
    assert experiment_data == edata_expected_with_output


# =============================================================================


@pytest.mark.parametrize(
    "input_data",
    ["test_input.csv", pd_input(), arr_input(), list_of_dicts_input()],
)
@pytest.mark.parametrize(
    "domain",
    [
        "test_domain.json",
        edata_domain_without_output(),
        domain_dictconfig_without_output(),
    ],
)
@pytest.mark.parametrize(
    "jobs", [edata_jobs_without_output(), "test_jobs.csv", None]
)
@pytest.mark.parametrize(
    "project_dir", ["./test_project", Path("./test_project")]
)
def test_experimentdata_creation_without_output(
    input_data,
    domain,
    jobs,
    project_dir,
    edata_expected_without_output,
    monkeypatch,
):
    def mock_read_csv(path, *args, **kwargs):
        if str(path) == "test_input.csv":
            return pd_input()
        elif str(path) == "test_output.csv":
            return pd_output()
        elif str(path) == "test_jobs.csv":
            return edata_jobs_without_output()
        raise ValueError(f"Unexpected file path: {path}")

    def mock_domain_from_file(path, *args, **kwargs):
        return edata_domain_without_output()

    monkeypatch.setattr(pd, "read_csv", mock_read_csv)
    monkeypatch.setattr(Domain, "from_file", mock_domain_from_file)
    mock_stat = MagicMock()
    mock_stat.st_size = 10  # Non Empty file

    with (
        patch.object(Path, "exists", return_value=True),
        patch.object(Path, "stat", return_value=mock_stat),
    ):
        experiment_data = ExperimentData(
            domain=domain,
            input_data=input_data,
            jobs=jobs,
            project_dir=project_dir,
        )

    if domain is None:
        experiment_data._domain = edata_domain_without_output()

    experiment_data = experiment_data.round(3)
    assert experiment_data == edata_expected_without_output
