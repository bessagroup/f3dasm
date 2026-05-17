"""Tests for strategy adapters and the default registry."""
#                                                                       Modules
# =============================================================================
from __future__ import annotations

# Third-party
import pytest

# Local
from f3dasm import ExperimentData
from f3dasm._src.datageneration.lookup import LookupDataGenerator
from f3dasm._src.optimization.agent_dataclasses import StrategySpec
from f3dasm._src.optimization.strategies import (
    default_registry,
    validate_params,
)
from f3dasm.design import Domain

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Elvis Aguero (elvis_alexander_aguero_vera@brown.edu)"
__credits__ = ["Elvis Aguero"]
__status__ = "Experimental"
# =============================================================================


def _build_pool_and_domain() -> (
    tuple[ExperimentData, Domain, LookupDataGenerator]
):
    """Construct a small 2D pool and a matching live ExperimentData shell."""

    domain = Domain()
    domain.add_float("x0", low=0.0, high=1.0)
    domain.add_float("x1", low=0.0, high=1.0)
    domain.add_output("y")
    domain.add_output("__turn")

    pool = ExperimentData(
        domain=domain,
        input_data=[
            {"x0": 0.1, "x1": 0.1},
            {"x0": 0.5, "x1": 0.5},
            {"x0": 0.9, "x1": 0.9},
            {"x0": 0.2, "x1": 0.8},
            {"x0": 0.8, "x1": 0.2},
        ],
        output_data=[
            {"y": 1.0},
            {"y": 2.0},
            {"y": 5.0},
            {"y": 3.0},
            {"y": 4.0},
        ],
    )
    live = ExperimentData(domain=domain)
    generator = LookupDataGenerator(pool, input_columns=["x0", "x1"])
    return live, domain, generator


def test_default_registry_contains_expected_strategies():
    adapters, specs = default_registry()
    assert set(adapters) == set(specs)
    assert (
        {"latin", "sobol", "random_uniform", "grid", "local_random"}
        <= set(adapters)
    )


def test_latin_adapter_writes_rows_with_turn_stamp():
    live, domain, gen = _build_pool_and_domain()
    adapters, _ = default_registry()

    outcome = adapters["latin"](
        n_steps=4,
        params={"seed": 42},
        domain=domain,
        data=live,
        data_generator=gen,
        objective_name="y",
        turn_id=7,
    )

    assert len(outcome.new_experiment_ids) == 4
    for idx in outcome.new_experiment_ids:
        assert live.data[idx]._output_data["__turn"] == 7
        assert "y" in live.data[idx]._output_data
    assert "Added 4 new rows" in outcome.summary


def test_local_random_respects_center_and_radius():
    live, domain, gen = _build_pool_and_domain()
    adapters, _ = default_registry()

    outcome = adapters["local_random"](
        n_steps=20,
        params={"center": {"x0": 0.5, "x1": 0.5}, "radius": 0.1, "seed": 0},
        domain=domain,
        data=live,
        data_generator=gen,
        objective_name="y",
        turn_id=3,
    )

    assert len(outcome.new_experiment_ids) == 20
    for idx in outcome.new_experiment_ids:
        x0 = live.data[idx]._input_data["x0"]
        x1 = live.data[idx]._input_data["x1"]
        # The box is [0.4, 0.6] x [0.4, 0.6] given radius=0.1.
        assert 0.4 <= x0 <= 0.6
        assert 0.4 <= x1 <= 0.6
        assert live.data[idx]._output_data["__turn"] == 3


def test_local_random_rejects_bad_radius():
    live, domain, gen = _build_pool_and_domain()
    adapters, _ = default_registry()

    with pytest.raises(ValueError):
        adapters["local_random"](
            n_steps=5,
            params={"center": {}, "radius": 1.5},
            domain=domain,
            data=live,
            data_generator=gen,
            objective_name="y",
            turn_id=0,
        )


def test_strategy_summary_includes_repeat_warning():
    live, domain, gen = _build_pool_and_domain()
    adapters, _ = default_registry()

    # Repeatedly target the same region so the lookup hits already-seen
    # pool entries.
    adapters["local_random"](
        n_steps=5,
        params={"center": {"x0": 0.5, "x1": 0.5}, "radius": 0.05, "seed": 1},
        domain=domain,
        data=live,
        data_generator=gen,
        objective_name="y",
        turn_id=0,
    )
    outcome = adapters["local_random"](
        n_steps=5,
        params={"center": {"x0": 0.5, "x1": 0.5}, "radius": 0.05, "seed": 2},
        domain=domain,
        data=live,
        data_generator=gen,
        objective_name="y",
        turn_id=1,
    )
    assert "WARNING" in outcome.summary


def test_validate_params_unknown_field():
    spec = StrategySpec(
        name="dummy",
        description="",
        parameters=default_registry()[1]["latin"].parameters,
    )
    error = validate_params(spec, {"unknown_field": 1})
    assert error is not None
    assert "Unknown parameter" in error


def test_validate_params_missing_required():
    spec = default_registry()[1]["local_random"]
    error = validate_params(spec, {"center": {}})
    assert error is not None
    assert "Missing required parameter" in error


def test_validate_params_accepts_valid():
    spec = default_registry()[1]["latin"]
    assert validate_params(spec, {"seed": 1}) is None
    assert validate_params(spec, {}) is None
