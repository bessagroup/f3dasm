import numpy as np
import pytest

import f3dasm

# WIP

@pytest.fixture(scope="module")
def design(dim):
    return f3dasm.make_nd_continuous_design(bounds=np.tile([-1.0, 1.0], (dim, 1)), dimensionality=dim)

@pytest.fixture(scope="module")
def sampler(design):
    return f3dasm.sampling.RandomUniform(design)

@pytest.fixture(scope="module")
def optimizer(design):
    return f3dasm.optimization.RandomSearch(data=f3dasm.Data(design))

@pytest.fixture(scope="module")
def function(dim):
    return f3dasm.functions.Levy(dimensionality=dim)

@pytest.fixture(scope="module")
def implementation_dict():
    return {}
