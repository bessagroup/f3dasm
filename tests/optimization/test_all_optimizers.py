from __future__ import annotations

import numpy as np
import pytest

from f3dasm import ExperimentData, create_sampler
from f3dasm._src.datageneration.datagenerator_factory import \
    create_datagenerator
from f3dasm._src.optimization.optimizer_factory import create_optimizer
from f3dasm import DataGenerator
from f3dasm._src.datageneration.benchmarkfunctions import BENCHMARK_FUNCTIONS
from f3dasm.design import make_nd_continuous_domain
from f3dasm._src.optimization.optimizer_factory import OPTIMIZERS


pytestmark = pytest.mark.smoke


@pytest.mark.parametrize("seed", [42])
@pytest.mark.parametrize("optimizer", list(OPTIMIZERS.keys()))
@pytest.mark.parametrize("data_generator", list(BENCHMARK_FUNCTIONS.keys()))
def test_all_optimizers_and_functions(
        seed: int, data_generator: str, optimizer: str,
        data: ExperimentData):
    _benchmark_function = create_datagenerator(
        data_generator=data_generator,
        output_names='y', seed=seed)

    _optimizer = create_optimizer(optimizer=optimizer)

    _benchmark_function.arm(data=data)

    samples = _benchmark_function.call(data=data)

    _optimizer.arm(data=samples)

    data1 = _optimizer.call(data=samples, data_generator=_benchmark_function,
                            )


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
