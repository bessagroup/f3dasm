import optuna  # noqa
import pytest

from f3dasm import ExperimentData, create_sampler, datagenerator
from f3dasm._src.optimization.optuna_implementations import OptunaOptimizer
from f3dasm.design import Domain


def test_optuna_optimizer():
    @datagenerator(output_names="y")
    def f(x):
        return x**2 + 1

    sampler = optuna.samplers.TPESampler(seed=42)

    optimizer = OptunaOptimizer(
        optuna_sampler=sampler,
    )

    domain = Domain()
    domain.add_float(name="x", low=-10, high=10)
    domain.add_output("y")
    data = ExperimentData(domain=domain)

    sampler = create_sampler("random")

    data = sampler.call(data, n_samples=1)

    f.arm(data)
    data = f.call(data)

    optimizer.arm(data, data_generator=f, input_name=None, output_name="y")

    data = optimizer.call(data, n_iterations=10)


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
