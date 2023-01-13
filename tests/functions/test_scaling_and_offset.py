from f3dasm.functions import FUNCTIONS_2D, get_functions
from f3dasm.base.function import Function
from typing import List

import numpy as np
import pytest

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize("function", FUNCTIONS_2D)
@pytest.mark.parametrize("seed", range(1))
def test_offset(function: Function, seed: int):

    dim = 2

    func: Function = function(
        seed=seed,
        dimensionality=dim,
        scale_bounds=np.tile([0.0, 1.0], (dim, 1)),
    )

    xmin = func._check_global_minimum()
    print(xmin)
    assert func.check_if_within_bounds(xmin)


@pytest.mark.parametrize("function", get_functions())
def test_check_global_minimum(function: Function):

    dim = 6
    if not function.is_dim_compatible(dim):
        dim = 4
        if not function.is_dim_compatible(dim):
            dim = 3
            if not function.is_dim_compatible(dim):
                dim = 2

    seed = 42
    func = function(noise=False, seed=seed, dimensionality=dim)
    x_min = func.get_global_minimum(dim)


@pytest.mark.parametrize("function", get_functions())
@pytest.mark.parametrize("scale_bounds_list", ([-1.0, 0.0], [-1.0, 1.0], [0.0, 1.0], [-3.0, 1.0]))
def test_scaling_1(function: Function, scale_bounds_list: List[float]):

    dim = 6
    if not function.is_dim_compatible(dim):
        dim = 4
        if not function.is_dim_compatible(dim):
            dim = 3
            if not function.is_dim_compatible(dim):
                dim = 2

    seed = np.random.randint(low=0, high=1e5)
    scale_bounds = np.tile(scale_bounds_list, (dim, 1))
    func = function(noise=False, seed=seed, scale_bounds=scale_bounds, dimensionality=dim)

    x = np.random.uniform(low=scale_bounds[0, 0], high=scale_bounds[0, 1], size=(1, func.dimensionality))

    assert func._scale_input(func._descale_input(x)) == pytest.approx(x)


@pytest.mark.parametrize("function", get_functions())
@pytest.mark.parametrize("scale_bounds_list", ([-1.0, 0.0], [-1.0, 1.0], [0.0, 1.0], [-3.0, 1.0]))
def test_scaling_2(function: Function, scale_bounds_list: List[float]):

    dim = 6
    if not function.is_dim_compatible(dim):
        dim = 4
        if not function.is_dim_compatible(dim):
            dim = 3
            if not function.is_dim_compatible(dim):
                dim = 2

    seed = np.random.randint(low=0, high=1e5)
    scale_bounds = np.tile(scale_bounds_list, (dim, 1))
    func = function(noise=False, seed=seed, scale_bounds=scale_bounds, dimensionality=dim)

    x = np.random.uniform(low=scale_bounds[0, 0], high=scale_bounds[0, 1], size=(1, dim))

    assert func._descale_input(func._scale_input(x)) == pytest.approx(x)


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
