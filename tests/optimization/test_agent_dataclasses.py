"""Tests for the agentic-f3dasm dataclasses module."""
#                                                                       Modules
# =============================================================================
from __future__ import annotations

# Standard
import dataclasses
import json

# Third-party
import pytest

# Local
from f3dasm._src.optimization.agent_dataclasses import (
    OutputSpec,
    ParameterSpec,
    ParamSignature,
    ProblemSchema,
    Strategy,
    StrategySpec,
    TurnRecord,
)

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Elvis Aguero (elvis_alexander_aguero_vera@brown.edu)"
__credits__ = ["Elvis Aguero"]
__status__ = "Experimental"
# =============================================================================


# Construction and immutability of frozen specs.
def test_parameter_spec_construction():
    spec = ParameterSpec(kind="continuous", bounds=(0.0, 1.0))
    assert spec.kind == "continuous"
    assert spec.bounds == (0.0, 1.0)
    assert spec.categories is None
    assert spec.log_scale is False


def test_parameter_spec_is_frozen():
    spec = ParameterSpec(kind="continuous", bounds=(0.0, 1.0))
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.kind = "categorical"


def test_output_spec_categorical():
    spec = OutputSpec(
        kind="categorical",
        categories=[0, 1, 2],
        description="three buckling modes",
    )
    assert spec.kind == "categorical"
    assert spec.categories == [0, 1, 2]


def test_problem_schema_construction():
    schema = ProblemSchema(
        variable_parameters={
            "x": ParameterSpec(kind="continuous", bounds=(0.0, 1.0))
        },
        forbidden_parameters=["const_p"],
        objective_name="y",
        objective_direction="maximize",
        output_columns={
            "y": OutputSpec(kind="continuous", description="objective")
        },
        physics_context="example",
    )
    assert schema.objective_direction == "maximize"
    assert "x" in schema.variable_parameters
    assert "const_p" in schema.forbidden_parameters


def test_strategy_spec_with_signatures():
    sig = {
        "n_samples": ParamSignature(
            type="int",
            required=True,
            default=None,
            description="samples to draw",
        )
    }
    spec = StrategySpec(
        name="latin", description="Latin hypercube", parameters=sig
    )
    assert spec.name == "latin"
    assert spec.parameters["n_samples"].required


def test_strategy_payload_construction():
    payload = Strategy(
        name="nelder_mead",
        n_steps=30,
        params={"x0": [0.5, 0.5]},
        intent="exploit the best coilable=1 region",
    )
    assert payload.n_steps == 30
    assert payload.intent.startswith("exploit")


# TurnRecord round-trip.
def test_turn_record_to_json_roundtrip():
    payload = Strategy(
        name="latin",
        n_steps=10,
        params={"n_samples": 10},
        intent="seed the search uniformly",
    )
    record = TurnRecord(
        turn_id=7,
        timestamp="2026-05-17T12:00:00",
        agent_name="strategizer",
        rationale="we have not yet covered the upper ratio_d region",
        emitted_strategy=payload,
    )

    line = record.to_json()
    parsed = json.loads(line)

    # Sanity-check the wire shape.
    assert parsed["turn_id"] == 7
    assert parsed["agent_name"] == "strategizer"
    assert parsed["emitted_strategy"]["name"] == "latin"
    assert parsed["result"] == "ok"
    assert parsed["experiment_ids"] == []

    restored = TurnRecord.from_json(line)
    assert restored.turn_id == 7
    assert restored.emitted_strategy == payload
    assert restored.strategy_calls == []
    assert restored.result == "ok"


def test_turn_record_implementer_with_strategy_calls():
    record = TurnRecord(
        turn_id=8,
        timestamp="2026-05-17T12:00:05",
        agent_name="implementer",
        rationale="ran latin(10), then nelder_mead(20)",
        strategy_calls=[
            ("latin", 10, {"n_samples": 10}),
            ("nelder_mead", 20, {"x0": [0.5, 0.5]}),
        ],
        experiment_ids=[100, 101, 102, 103, 104],
    )

    restored = TurnRecord.from_json(record.to_json())
    assert restored.strategy_calls[0] == ("latin", 10, {"n_samples": 10})
    assert restored.strategy_calls[1][0] == "nelder_mead"
    assert restored.experiment_ids == [100, 101, 102, 103, 104]


def test_turn_record_error_result_persists():
    record = TurnRecord(
        turn_id=9,
        timestamp="2026-05-17T12:00:10",
        agent_name="strategizer",
        rationale="",
        result="error",
    )
    restored = TurnRecord.from_json(record.to_json())
    assert restored.result == "error"
    assert restored.rationale == ""
    assert restored.emitted_strategy is None
