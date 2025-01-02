from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from f3dasm import ExperimentData
from f3dasm._src.design.parameter import (CategoricalParameter,
                                          ContinuousParameter,
                                          DiscreteParameter)
from f3dasm.design import Domain, make_nd_continuous_domain

SEED = 42


@pytest.fixture(scope="package")
def seed() -> int:
    return SEED


@pytest.fixture(scope="package")
def domain() -> Domain:

    space = {
        'x1': ContinuousParameter(-5.12, 5.12),
        'x2': DiscreteParameter(-3, 3),
        'x3': CategoricalParameter(["red", "green", "blue"])
    }

    return Domain(input_space=space)


@pytest.fixture(scope="package")
def domain_continuous() -> Domain:
    return make_nd_continuous_domain(bounds=np.array([[0., 1.], [0., 1.], [0., 1.]]), dimensionality=3)


@pytest.fixture(scope="package")
def experimentdata(domain: Domain) -> ExperimentData:
    e_data = ExperimentData(domain)
    e_data.sample(sampler='random', n_samples=10, seed=SEED)
    return e_data


@pytest.fixture(scope="package")
def experimentdata2(domain: Domain) -> ExperimentData:
    return ExperimentData.from_sampling(sampler='random', domain=domain, n_samples=10, seed=SEED)


@pytest.fixture(scope="package")
def experimentdata_continuous(domain_continuous: Domain) -> ExperimentData:
    return ExperimentData.from_sampling(sampler='random', domain=domain_continuous, n_samples=10, seed=SEED)


@pytest.fixture(scope="package")
def experimentdata_expected() -> ExperimentData:
    domain_continuous = make_nd_continuous_domain(
        bounds=np.array([[0., 1.], [0., 1.], [0., 1.]]), dimensionality=3)
    data = ExperimentData.from_sampling(
        sampler='random', domain=domain_continuous, n_samples=10, seed=SEED)
    for es, output in zip(data, np.zeros((10, 1))):
        es.store(name='y', object=float(output))
        data._set_experiment_sample(es)
    data.add(input_data=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
             output_data=np.array([[0.0], [0.0]]), domain=domain_continuous)

    data._input_data.round(6)
    # data._input_data.data = [[round(num, 6) if isinstance(
    #     num, float) else num for num in sublist]
    #     for sublist in data._input_data.data]
    return data


@pytest.fixture(scope="package")
def experimentdata_expected_no_output() -> ExperimentData:
    domain_continuous = make_nd_continuous_domain(
        bounds=np.array([[0., 1.], [0., 1.], [0., 1.]]), dimensionality=3)
    data = ExperimentData.from_sampling(
        sampler='random', domain=domain_continuous, n_samples=10, seed=SEED)
    data.add(input_data=np.array(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]), domain=domain_continuous)

    data._input_data.round(6)
    # data._input_data.data = [[round(num, 6) if isinstance(
    #     num, float) else num for num in sublist]
    #     for sublist in data._input_data.data]
    return data


@pytest.fixture(scope="package")
def experimentdata_expected_only_domain() -> ExperimentData:
    domain_continuous = make_nd_continuous_domain(
        bounds=np.array([[0., 1.], [0., 1.], [0., 1.]]), dimensionality=3)
    return ExperimentData(domain=domain_continuous)


@pytest.fixture(scope="package")
def numpy_array(domain_continuous: Domain) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    return rng.random((10, len(domain_continuous)))


@pytest.fixture(scope="package")
def numpy_output_array(domain_continuous: Domain) -> np.ndarray:
    return np.zeros((10, 1))


@pytest.fixture(scope="package")
def xarray_dataset(domain_continuous: Domain) -> xr.Dataset:
    rng = np.random.default_rng(SEED)
    # np.random.seed(SEED)
    input_data = rng.random((10, len(domain_continuous)))
    input_names = domain_continuous.input_names

    output_data = pd.DataFrame()
    output_names = output_data.columns.to_list()

    return xr.Dataset({'input': xr.DataArray(input_data, dims=['iterations', 'input_dim'], coords={
        'iterations': range(len(input_data)), 'input_dim': input_names}),
        'output': xr.DataArray(output_data, dims=['iterations', 'output_dim'], coords={
            'iterations': range(len(output_data)), 'output_dim': output_names})})


@pytest.fixture(scope="package")
def pandas_dataframe(domain_continuous: Domain) -> pd.DataFrame:
    # np.random.seed(SEED)
    rng = np.random.default_rng(SEED)
    return pd.DataFrame(rng.random((10, len(domain_continuous))),
                        columns=domain_continuous.input_names)


@pytest.fixture(scope="package")
def continuous_parameter() -> ContinuousParameter:
    return ContinuousParameter(lower_bound=0., upper_bound=1.)
