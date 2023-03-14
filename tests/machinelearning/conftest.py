import numpy as np
import pytest

from f3dasm import make_nd_continuous_design
from f3dasm.design.design import DesignSpace

SEED = 42
DIMENSIONALITY = 5


@pytest.fixture(scope="package")
def design() -> DesignSpace:
    return make_nd_continuous_design(bounds=np.tile([-1.0, 1.0], (DIMENSIONALITY, 1)), dimensionality=DIMENSIONALITY)


@pytest.fixture(scope="package")
def design_2d() -> DesignSpace:
    return make_nd_continuous_design(bounds=np.tile([-1.0, 1.0], (2, 1)), dimensionality=2)


@pytest.fixture(scope="package")
def design_7d() -> DesignSpace:
    return make_nd_continuous_design(bounds=np.tile([-1.0, 1.0], (7, 1)), dimensionality=7)
