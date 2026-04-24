import optuna  # noqa
import pytest

from f3dasm import ExperimentData, create_sampler, datagenerator
from f3dasm._src.optimization.optuna_implementations import (
    OptunaUpdateStep,
    optuna_optimizer,
    tpesampler,
)
from f3dasm.design import Domain


def test_optuna_update_step_loop():
    @datagenerator(output_names="y")
    def f(x):
        return x**2 + 1

    sampler = optuna.samplers.TPESampler(seed=42)
    update_step = OptunaUpdateStep(optuna_sampler=sampler, output_name="y")

    domain = Domain()
    domain.add_float(name="x", low=-10, high=10)
    domain.add_output("y")
    data = ExperimentData(domain=domain)

    data = create_sampler("random").call(data, n_samples=1)
    f.arm(data)
    data = f.call(data)

    n_iterations = 10
    starting_rows = len(data)
    loop = (update_step >> f).loop(n_iterations)
    loop.arm(data)
    data = loop.call(data)

    assert len(data) == starting_rows + n_iterations


def test_optuna_optimizer_factory_returns_update_step():
    step = optuna_optimizer(
        optuna_sampler=optuna.samplers.TPESampler(), output_name="y"
    )
    assert isinstance(step, OptunaUpdateStep)


def test_tpesampler_factory_returns_update_step():
    step = tpesampler(output_name="y")
    assert isinstance(step, OptunaUpdateStep)


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
