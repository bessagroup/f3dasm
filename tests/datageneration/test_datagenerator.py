from typing import Callable

import numpy as np
import pytest

from f3dasm import (
    DataGenerator,
    ExperimentData,
    create_datagenerator,
)
from f3dasm._src.datageneration.datagenerator_factory import datagenerator
from f3dasm.design import Domain

pytestmark = pytest.mark.smoke


def test_convert_function(
    experiment_data: ExperimentData, function_1: Callable
):
    fn = datagenerator(output_names=["y0", "y1"])(function_1)

    assert isinstance(fn, DataGenerator)

    _ = fn.call(experiment_data, s=103)


def test_convert_function2(
    experiment_data: ExperimentData, function_2: Callable
):
    fn = datagenerator(output_names=["y0", "y1"])(function_2)

    assert isinstance(fn, DataGenerator)

    _ = fn.call(experiment_data)


@pytest.mark.parametrize("mode", ["sequential", "parallel"])
def test_parallelization(mode, tmp_path):
    domain = Domain()
    domain.add_parameter("x")
    domain.add_output("y")
    experiment_data = ExperimentData(
        domain=domain, input_data=[{"x": np.array([0.5, 0.5])}]
    )

    experiment_data.set_project_dir(tmp_path, in_place=True)

    func = create_datagenerator(data_generator="ackley", output_names="y")

    func.arm(data=experiment_data)

    experiment_data = func.call(data=experiment_data, mode=mode)


def test_invalid_parallelization_mode():
    domain = Domain()
    domain.add_parameter("x")
    domain.add_output("y")
    experiment_data = ExperimentData(
        domain=domain, input_data=[{"x": np.array([0.5, 0.5])}]
    )

    func = create_datagenerator(data_generator="ackley", output_names="y")

    with pytest.raises(ValueError):
        experiment_data = func.call(data=experiment_data, mode="invalid")


# ====== signature validation (#268) =========================================


def test_decorated_function_with_mismatched_arg_raises_on_call():
    """User function declares `seed`, Domain provides `x`/`z` -- arm must
    raise immediately rather than letting `_run_sample` fail with the
    confusing `cannot unpack non-iterable NoneType object` error.

    This is the second repro from issue #268 (renamed argument).
    """
    domain = Domain()
    domain.add_float("x", low=0.0, high=100.0)
    domain.add_float("z", low=0.0, high=100.0)
    domain.add_output("y")

    @datagenerator(output_names="y")
    def fn(seed):
        return seed

    ed = ExperimentData(domain=domain, input_data=[{"x": 1.0, "z": 2.0}])
    with pytest.raises(ValueError, match=r"required argument\(s\) \['seed'\]"):
        fn.call(data=ed)


def test_decorated_function_with_extra_domain_arg_passes():
    """Domain may legitimately carry parameters the function doesn't
    consume (e.g. metadata kept for downstream blocks). The validation
    should only flag function args that have no Domain backing."""
    domain = Domain()
    domain.add_float("x", low=0.0, high=100.0)
    domain.add_float("z", low=0.0, high=100.0)
    domain.add_output("y")

    @datagenerator(output_names="y")
    def fn(x):
        return x * 2

    ed = ExperimentData(domain=domain, input_data=[{"x": 1.0, "z": 2.0}])
    # Should not raise.
    fn.arm(data=ed)


def test_decorated_function_with_defaulted_arg_passes():
    """Arguments with defaults are optional, so a missing Domain entry
    for them must not trigger the validation."""
    domain = Domain()
    domain.add_float("x", low=0.0, high=1.0)
    domain.add_output("y")

    @datagenerator(output_names="y")
    def fn(x, multiplier=2.0):
        return x * multiplier

    ed = ExperimentData(domain=domain, input_data=[{"x": 1.0}])
    fn.arm(data=ed)  # no raise
    fn.call(data=ed)  # no raise either
