from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from omegaconf import DictConfig

from f3dasm import ExperimentData
from f3dasm.design import Domain, make_nd_continuous_domain

SEED = 42

pytestmark = pytest.mark.smoke


def domain_dictconfig():
    config_dict = {"input": {"x0": {"type": "float", "low": 0.0, "high": 1.0},
                   "x1": {"type": "float", "low": 0.0, "high": 1.0},
                   "x2": {"type": "float", "low": 0.0, "high": 1.0}},

                   "output": {'y': {}}
                   }

    return DictConfig(config_dict)


def test_project_dir_false_data():
    with pytest.raises(TypeError):
        ExperimentData(project_dir=0)


def experiment_data() -> ExperimentData:
    domain = make_nd_continuous_domain(
        bounds=[[0., 1.], [0., 1.], [0., 1.]])

    data = ExperimentData.from_sampling(
        sampler='random', domain=domain, n_samples=10, seed=SEED
    )

    data += ExperimentData(
        input_data=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        domain=data.domain)

    def f(*args, **kwargs):
        return 0.0

    data.evaluate(data_generator=f, output_names=['y'])
    data.round(3)

    data.set_project_dir('./test_project')
    return data


@ pytest.fixture(scope="package")
def edata_expected() -> ExperimentData:
    domain = make_nd_continuous_domain(
        bounds=[[0., 1.], [0., 1.], [0., 1.]])

    data = ExperimentData.from_sampling(
        sampler='random', domain=domain, n_samples=10, seed=SEED
    )

    data += ExperimentData(
        input_data=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
        domain=data.domain)

    def f(*args, **kwargs):
        return 0.0

    data.evaluate(data_generator=f, output_names=['y'])
    data.round(3)

    data.set_project_dir('./test_project')

    return data


def edata_domain() -> Domain:
    return experiment_data().domain


def edata_jobs() -> pd.Series:
    return experiment_data().jobs


def arr_input():
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
    ]).round(3)


def list_of_dicts_input():
    return [
        {'x0': 0.77395605, 'x1': 0.43887844, 'x2': 0.85859792},
        {'x0': 0.69736803, 'x1': 0.09417735, 'x2': 0.97562235},
        {'x0': 0.7611397, 'x1': 0.78606431, 'x2': 0.12811363},
        {'x0': 0.45038594, 'x1': 0.37079802, 'x2': 0.92676499},
        {'x0': 0.64386512, 'x1': 0.82276161, 'x2': 0.4434142},
        {'x0': 0.22723872, 'x1': 0.55458479, 'x2': 0.06381726},
        {'x0': 0.82763117, 'x1': 0.6316644, 'x2': 0.75808774},
        {'x0': 0.35452597, 'x1': 0.97069802, 'x2': 0.89312112},
        {'x0': 0.7783835, 'x1': 0.19463871, 'x2': 0.466721},
        {'x0': 0.04380377, 'x1': 0.15428949, 'x2': 0.68304895},
        {'x0': 0.000000, 'x1': 0.000000, 'x2': 0.000000},
        {'x0': 1.000000, 'x1': 1.000000, 'x2': 1.000000},
    ]


def list_of_dicts_output():
    return [
        {'y': 0.0},
        {'y': 0.0},
        {'y': 0.0},
        {'y': 0.0},
        {'y': 0.0},
        {'y': 0.0},
        {'y': 0.0},
        {'y': 0.0},
        {'y': 0.0},
        {'y': 0.0},
        {'y': 0.0},
        {'y': 0.0},
    ]


def pd_input():
    return pd.DataFrame(arr_input(), columns=['x0', 'x1', 'x2'])


def arr_output():
    return np.array([[0.],
                     [0.],
                     [0.],
                     [0.],
                     [0.],
                     [0.],
                     [0.],
                     [0.],
                     [0.],
                     [0.],
                     [0.],
                     [0.]])


def pd_output():
    return pd.DataFrame(arr_output(), columns=['y'])


@ pytest.mark.parametrize("input_data", ["test_input.csv", pd_input(), arr_input(), list_of_dicts_input()])
@ pytest.mark.parametrize("output_data", ["test_output.csv", pd_output(), arr_output(), list_of_dicts_output()])
@ pytest.mark.parametrize("domain", ["test_domain.pkl", edata_domain(), None, domain_dictconfig()])
@ pytest.mark.parametrize("jobs", [edata_jobs(), "test_jobs.csv", None])
@ pytest.mark.parametrize("project_dir", ["./test_project", Path("./test_project")])
def test_experimentdata_creation(
        input_data, output_data, domain, jobs, project_dir, edata_expected, monkeypatch):

    def mock_read_csv(path, *args, **kwargs):
        if str(path) == "test_input.csv":
            return pd_input()
        elif str(path) == "test_output.csv":
            return pd_output()
        elif str(path) == "test_jobs.csv":
            return edata_jobs()
        raise ValueError(f"Unexpected file path: {path}")

    def mock_domain_from_file(path, *args, **kwargs):
        return edata_domain()

    monkeypatch.setattr(pd, 'read_csv', mock_read_csv)
    monkeypatch.setattr(Domain, 'from_file', mock_domain_from_file)

    experiment_data = ExperimentData(domain=domain, input_data=input_data,
                                     output_data=output_data, jobs=jobs,
                                     project_dir=project_dir)

    if domain is None:
        experiment_data.domain = edata_domain()

    experiment_data.round(3)
    # assert experiment_data.domain == edata_expected.domain
    # assert experiment_data.data == edata_expected.data
    assert experiment_data == edata_expected


def test_experiment_data_from_yaml_sampling():
    domain = make_nd_continuous_domain(
        bounds=[[0., 1.], [0., 1.], [0., 1.]])

    data_expected = ExperimentData.from_sampling(
        sampler='random', domain=domain, n_samples=10, seed=SEED
    )

    dict_config = DictConfig({'from_sampling': {
        'sampler': 'random',
        'domain': {"x0": {"type": "float", "low": 0.0, "high": 1.0},
                   "x1": {"type": "float", "low": 0.0, "high": 1.0},
                   "x2": {"type": "float", "low": 0.0, "high": 1.0},
                   },
        'n_samples': 10,
        'seed': SEED
    }})

    data = ExperimentData.from_yaml(dict_config)

    assert data.domain == data_expected.domain
