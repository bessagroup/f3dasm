import numpy as np
import pytest

from f3dasm import make_nd_continuous_domain
from f3dasm.datageneration.functions import Ackley
from f3dasm.datageneration.datagenerator import DataGenerator
from f3dasm.design.domain import Domain
from f3dasm.optimization import Optimizer, RandomSearch
from f3dasm.run_optimization import (OptimizationResult,
                                     run_multiple_realizations)
from f3dasm.sampling import LatinHypercube, Sampler

SEED = 42
DIMENSIONALITY = 5


@pytest.fixture(scope="package")
def design() -> Domain:
    return make_nd_continuous_domain(bounds=np.tile([-1.0, 1.0], (DIMENSIONALITY, 1)), dimensionality=DIMENSIONALITY)


@pytest.fixture(scope="package")
def design_7d() -> Domain:
    return make_nd_continuous_domain(bounds=np.tile([-1.0, 1.0], (7, 1)), dimensionality=7)


@pytest.fixture(scope="package")
def sampler(design: Domain) -> Sampler:
    return LatinHypercube(domain=design, seed=SEED)


@pytest.fixture(scope="package")
def function(design: Domain) -> DataGenerator:
    return Ackley(dimensionality=DIMENSIONALITY, scale_bounds=design.get_bounds(), seed=SEED)


@pytest.fixture(scope="package")
def number_of_samples() -> int:
    return 30


@pytest.fixture(scope="package")
def optimizer(sampler: Sampler, number_of_samples: int) -> Optimizer:
    samples = sampler.get_samples(numsamples=number_of_samples)
    return RandomSearch(domain=samples.domain, seed=SEED)


@pytest.fixture(scope="package")
def optimizationresults(optimizer: Optimizer, sampler: Sampler,
                        function: DataGenerator, number_of_samples: int) -> OptimizationResult:
    args = {'optimizer': optimizer,
            'data_generator': function,
            'sampler': sampler,
            'iterations': 1000,
            'realizations': 5,
            'number_of_samples': number_of_samples,
            'parallelization': True,
            'seed': SEED
            }
    return run_multiple_realizations(**args)
