from typing import List

import numpy as np
import pytest

from f3dasm import ExperimentData
from f3dasm._src.datageneration.datagenerator_factory import \
    create_datagenerator
from f3dasm._src.datageneration.functions import (FUNCTIONS, Function,
                                                  get_function_classes)
from f3dasm.design import make_nd_continuous_domain

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize("function", get_function_classes(d=2))
@pytest.mark.parametrize("seed", range(1))
def test_offset(function: Function, seed: int):

    dim = 2
    domain = make_nd_continuous_domain(bounds=np.tile([0.0, 1.0], (dim, 1)))
    func: Function = function(
        seed=seed,
        scale_bounds=domain.get_bounds(),
    )

    func.arm(data=ExperimentData(domain=domain))

    xmin = func._get_global_minimum_for_offset_calculation()

    assert func.check_if_within_bounds(xmin, bounds=domain.get_bounds())


@pytest.mark.parametrize("function", FUNCTIONS)
def test_check_global_minimum(function: str):

    _func = create_datagenerator(
        data_generator=function)

    dim = 6
    domain = make_nd_continuous_domain(bounds=np.tile(
        [-1.0, 1.0], (dim, 1)), dimensionality=dim)
    if not _func.is_dim_compatible(d=dim):
        dim = 4
        domain = make_nd_continuous_domain(bounds=np.tile(
            [-1.0, 1.0], (dim, 1)), dimensionality=dim)
        if not _func.is_dim_compatible(d=dim):
            dim = 3
            domain = make_nd_continuous_domain(bounds=np.tile(
                [-1.0, 1.0], (dim, 1)), dimensionality=dim)
            if not _func.is_dim_compatible(d=dim):
                dim = 2
                domain = make_nd_continuous_domain(bounds=np.tile(
                    [-1.0, 1.0], (dim, 1)), dimensionality=dim)

    seed = 42
    func = create_datagenerator(
        data_generator=function, seed=seed)
    func.arm(data=ExperimentData(domain=domain))
    _ = func.get_global_minimum(dim)


@pytest.mark.parametrize("function", FUNCTIONS)
@pytest.mark.parametrize("scale_bounds_list",
                         ([-1.0, 0.0], [-1.0, 1.0], [0.0, 1.0], [-3.0, 1.0]))
def test_scaling_1(function: str, scale_bounds_list: List[float]):

    _func = create_datagenerator(
        data_generator=function)

    dim = 6
    domain = make_nd_continuous_domain(bounds=np.tile(
        [-1.0, 1.0], (dim, 1)), dimensionality=dim)
    if not _func.is_dim_compatible(d=dim):
        dim = 4
        domain = make_nd_continuous_domain(bounds=np.tile(
            [-1.0, 1.0], (dim, 1)), dimensionality=dim)
        if not _func.is_dim_compatible(d=dim):
            dim = 3
            domain = make_nd_continuous_domain(bounds=np.tile(
                [-1.0, 1.0], (dim, 1)), dimensionality=dim)
            if not _func.is_dim_compatible(d=dim):
                dim = 2
                domain = make_nd_continuous_domain(bounds=np.tile(
                    [-1.0, 1.0], (dim, 1)), dimensionality=dim)

    seed = np.random.randint(low=0, high=1e5)
    scale_bounds = np.tile(scale_bounds_list, (dim, 1))
    domain = make_nd_continuous_domain(bounds=np.tile(
        [-1.0, 1.0], (dim, 1)), dimensionality=dim)
    func = create_datagenerator(
        data_generator=function, seed=seed, scale_bounds=scale_bounds)
    func.arm(data=ExperimentData(domain=domain))
    x = np.random.uniform(
        low=scale_bounds[0, 0], high=scale_bounds[0, 1],
        size=(1, func.dimensionality))

    assert func._retrieve_original_input(
        func.augmentor.augment_input(x)) == pytest.approx(x)


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
