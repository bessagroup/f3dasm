from typing import List
import numpy as np
import pytest

pytestmark = pytest.mark.smoke

from f3dasm.base.function import Function
from f3dasm.functions import get_functions


@pytest.mark.parametrize("function", get_functions(d=6) + get_functions(d=3) + get_functions(d=4) + get_functions(d=2))
@pytest.mark.parametrize("seed", range(20))
def test_offset(function: Function, seed: int):
    func: Function = function(
        noise=False,
        seed=seed,
        dimensionality=function.dimensionality,
        scale_bounds=np.tile([0.0, 1.0], (function.dimensionality, 1)),
    )
    func._create_offset()
    xmin = func._check_global_minimum()
    assert func.check_if_within_bounds(xmin)


@pytest.mark.parametrize("function", get_functions(d=2))
def test_check_global_minimum(function: Function):
    seed = 42
    func = function(noise=False, seed=seed, dimensionality=function.dimensionality)
    x_min = func.get_global_minimum(function.dimensionality)


@pytest.mark.parametrize("function", get_functions(d=2))
@pytest.mark.parametrize("scale_bounds_list", ([-1.0, 0.0], [-1.0, 1.0], [0.0, 1.0], [-3.0, 1.0]))
def test_scaling_1(function: Function, scale_bounds_list: List[float]):
    seed = np.random.randint(low=0, high=1e5)
    scale_bounds = np.tile(scale_bounds_list, (function.dimensionality, 1))
    func = function(noise=False, seed=seed, scale_bounds=scale_bounds, dimensionality=function.dimensionality)

    x = np.random.uniform(low=scale_bounds[0, 0], high=scale_bounds[0, 1], size=(1, func.dimensionality))

    assert func._scale_input(func._descale_input(x)) == pytest.approx(x)


@pytest.mark.parametrize("function", get_functions(d=2))
@pytest.mark.parametrize("scale_bounds_list", ([-1.0, 0.0], [-1.0, 1.0], [0.0, 1.0], [-3.0, 1.0]))
def test_scaling_2(function: Function, scale_bounds_list: List[float]):
    seed = np.random.randint(low=0, high=1e5)
    scale_bounds = np.tile(scale_bounds_list, (function.dimensionality, 1))
    func = function(noise=False, seed=seed, scale_bounds=scale_bounds, dimensionality=function.dimensionality)

    x = np.random.uniform(low=scale_bounds[0, 0], high=scale_bounds[0, 1], size=(1, func.dimensionality))

    assert func._descale_input(func._scale_input(x)) == pytest.approx(x)


# @pytest.mark.parametrize("function", get_functions(d=2) + get_functions(d=6) + get_functions(d=3) + get_functions(d=4))
# def test_global_min_within_window(function: Function):
#     seed = np.random.randint(low=0, high=1e5)
#     func = function(noise=False, seed=seed, dimensionality=function.dimensionality)

#     assert func.check_if_within_bounds(func.get_global_minimum(func.dimensionality)[0])


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
