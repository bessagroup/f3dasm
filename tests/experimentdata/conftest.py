
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from omegaconf import OmegaConf

from f3dasm.datageneration.datagenerator import DataGenerator
from f3dasm.datageneration.functions import Ackley
from f3dasm.design.domain import Domain, make_nd_continuous_domain
from f3dasm.design.experimentdata import ExperimentData
from f3dasm.design.parameter import (CategoricalParameter, ContinuousParameter,
                                     DiscreteParameter)
from f3dasm.sampling import RandomUniform

SEED = 42


@pytest.fixture(scope="package")
def seed() -> int:
    return SEED


@pytest.fixture(scope="package")
def domain() -> Domain:

    input_space = {
        'x1': ContinuousParameter(-5.12, 5.12),
        'x2': DiscreteParameter(-3, 3),
        'x3': CategoricalParameter(["red", "green", "blue"])
    }

    return Domain(input_space=input_space)


@pytest.fixture(scope="package")
def domain_continuous() -> Domain:
    return make_nd_continuous_domain(bounds=np.array([[0., 1.], [0., 1.], [0., 1.]]), dimensionality=3)


@pytest.fixture(scope="package")
def experimentdata(domain: Domain) -> ExperimentData:
    sampler = RandomUniform(domain=domain, number_of_samples=10, seed=SEED)
    return ExperimentData.from_sampling(sampler)


@pytest.fixture(scope="package")
def experimentdata2(domain: Domain) -> ExperimentData:
    sampler = RandomUniform(domain=domain, number_of_samples=10, seed=SEED+1)
    return ExperimentData.from_sampling(sampler)


@pytest.fixture(scope="package")
def experimentdata_continuous(domain_continuous: Domain) -> ExperimentData:
    sampler = RandomUniform(domain=domain_continuous, number_of_samples=10, seed=SEED)
    return ExperimentData.from_sampling(sampler)


@pytest.fixture(scope="package")
def numpy_array(domain_continuous: Domain) -> np.ndarray:
    np.random.seed(SEED)
    return np.random.rand(10, len(domain_continuous))


@pytest.fixture(scope="package")
def numpy_output_array(domain_continuous: Domain) -> np.ndarray:
    return np.zeros((10, 1))


@pytest.fixture(scope="package")
def xarray_dataset(domain_continuous: Domain) -> xr.Dataset:
    np.random.seed(SEED)
    input_data = np.random.rand(10, len(domain_continuous))
    input_names = domain_continuous.names

    output_data = pd.DataFrame()
    output_names = output_data.columns.to_list()

    return xr.Dataset({'input': xr.DataArray(input_data, dims=['iterations', 'input_dim'], coords={
        'iterations': range(len(input_data)), 'input_dim': input_names}),
        'output': xr.DataArray(output_data, dims=['iterations', 'output_dim'], coords={
            'iterations': range(len(output_data)), 'output_dim': output_names})})


@pytest.fixture(scope="package")
def pandas_dataframe(domain_continuous: Domain) -> pd.DataFrame:
    np.random.seed(SEED)
    return pd.DataFrame(np.random.rand(10, len(domain_continuous)), columns=domain_continuous.names)

# Define test data


@pytest.fixture(scope="package")
def config_yaml() -> OmegaConf:
    # Define a sample configuration for testing
    config_dict = {
        "experimentdata": {
            "from_sampling": {"_target_": "f3dasm.sampling.RandomUniform", "number_of_samples": 10, "seed": SEED},
            "from_csv": {"input_filepath": "experimentdata_data.csv", "output_filepath": "experimentdata_output.csv"},
            "from_file": {"filepath": "experimentdata"},
            # Add other options as needed for testing
        },
        "domain": {
            "input_space": {"x1": {"_target_": "f3dasm.ContinuousParameter", "lower": -5.12, "upper": 5.12},
                            "x2": {"_target_": "f3dasm.DiscreteParameter", "lower": -3, "upper": 3},
                            "x3": {"_target_": "f3dasm.CategoricalParameter", "categories": ["red", "green", "blue"]}}}
    }

    return OmegaConf.create(config_dict)


@pytest.fixture(scope="package")
def continuous_parameter() -> ContinuousParameter:
    return ContinuousParameter(lower_bound=0., upper_bound=1.)


@pytest.fixture(scope="package")
def func() -> DataGenerator:
    return Ackley(dimensionality=3, scale_bounds=np.array([[0., 1.], [0., 1.], [0., 1.]]), seed=SEED)
