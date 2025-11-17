from __future__ import annotations

import numpy as np
import pytest

from f3dasm import ExperimentData, datagenerator
from f3dasm._src.datageneration.benchmarkfunctions import BENCHMARK_FUNCTIONS
from f3dasm._src.datageneration.datagenerator_factory import (
    create_datagenerator,
)
from f3dasm._src.optimization.optimizer_factory import (
    OPTIMIZERS,
    create_optimizer,
)

pytestmark = pytest.mark.smoke


@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize("optimizer", ["cg", "lbfgsb", "neldermead"])
@pytest.mark.parametrize("data_generator", list(BENCHMARK_FUNCTIONS.keys()))
def test_scipyminimize(
    seed: int, data_generator: str, optimizer: str, data: ExperimentData
):
    _benchmark_function = create_datagenerator(
        data_generator=data_generator, output_names="y", seed=seed
    )

    _optimizer = create_optimizer(optimizer=optimizer)

    _benchmark_function.arm(data=data)

    samples = _benchmark_function.call(data=data)

    _optimizer.arm(
        data=samples,
        data_generator=_benchmark_function,
        input_name="x",
        output_name="y",
    )

    data1 = _optimizer.call(
        data=samples,
        data_generator=_benchmark_function,
    )


@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize("optimizer", ["tpesampler"])
def test_tpesampler(seed: int, optimizer: str, data2: ExperimentData):
    @datagenerator(output_names="y")
    def f(x0):
        return x0**2

    f.arm(data2)
    data = f.call(data2)

    optimizer = create_optimizer(optimizer=optimizer)
    optimizer.arm(data, data_generator=f, output_name="y")

    data = optimizer.call(data, n_iterations=10)


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
