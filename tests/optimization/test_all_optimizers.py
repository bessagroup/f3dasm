from __future__ import annotations

import pytest

from f3dasm import ExperimentData, datagenerator
from f3dasm._src.datageneration.benchmarkfunctions import BENCHMARK_FUNCTIONS
from f3dasm._src.datageneration.datagenerator_factory import (
    create_datagenerator,
)
from f3dasm.optimization import cg, lbfgsb, nelder_mead, tpesampler

pytestmark = pytest.mark.smoke

SCIPY_FACTORIES = {"cg": cg, "lbfgsb": lbfgsb, "neldermead": nelder_mead}


@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize("optimizer", list(SCIPY_FACTORIES.keys()))
@pytest.mark.parametrize("data_generator", list(BENCHMARK_FUNCTIONS.keys()))
def test_scipyminimize(
    seed: int, data_generator: str, optimizer: str, data: ExperimentData
):
    _benchmark_function = create_datagenerator(
        data_generator=data_generator, output_names="y", seed=seed
    )

    _benchmark_function.arm(data=data)
    samples = _benchmark_function.call(data=data)

    _optimizer = SCIPY_FACTORIES[optimizer](
        data_generator=_benchmark_function,
        output_name="y",
        input_name="x",
    )
    _optimizer.arm(data=samples)
    _ = _optimizer.call(data=samples)


@pytest.mark.parametrize("seed", [42])
def test_tpesampler(seed: int, data2: ExperimentData):
    @datagenerator(output_names="y")
    def f(x0):
        return x0**2

    f.arm(data2)
    data = f.call(data2)

    step = (tpesampler(output_name="y") >> f).loop(10)
    step.arm(data)
    data = step.call(data)


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
