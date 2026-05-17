"""Tests for the MVP artifact producers."""
#                                                                       Modules
# =============================================================================
from __future__ import annotations

# Local
from f3dasm import ExperimentData
from f3dasm._src.optimization.agent_dataclasses import (
    OutputSpec,
    ParameterSpec,
    ProblemSchema,
    TurnRecord,
)
from f3dasm._src.optimization.artifact_producers import (
    coverage_summary,
    last_5_rationales,
    objective_summary,
)
from f3dasm.design import Domain

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Elvis Aguero (elvis_alexander_aguero_vera@brown.edu)"
__credits__ = ["Elvis Aguero"]
__status__ = "Experimental"
# =============================================================================


def _schema() -> ProblemSchema:
    return ProblemSchema(
        variable_parameters={
            "x0": ParameterSpec(kind="continuous", bounds=(0.0, 1.0)),
            "x1": ParameterSpec(kind="continuous", bounds=(0.0, 1.0)),
        },
        forbidden_parameters=[],
        objective_name="y",
        objective_direction="maximize",
        output_columns={
            "y": OutputSpec(kind="continuous", description="objective"),
            "label": OutputSpec(
                kind="categorical",
                categories=[0, 1],
                description="binary class",
            ),
        },
        physics_context="dummy",
    )


def _empty_data() -> ExperimentData:
    domain = Domain()
    domain.add_float("x0", low=0.0, high=1.0)
    domain.add_float("x1", low=0.0, high=1.0)
    domain.add_output("y")
    domain.add_output("label")
    domain.add_output("__turn")
    return ExperimentData(domain=domain)


def _populated_data() -> ExperimentData:
    domain = Domain()
    domain.add_float("x0", low=0.0, high=1.0)
    domain.add_float("x1", low=0.0, high=1.0)
    domain.add_output("y")
    domain.add_output("label")
    domain.add_output("__turn")
    return ExperimentData(
        domain=domain,
        input_data=[
            {"x0": 0.1, "x1": 0.1},
            {"x0": 0.5, "x1": 0.5},
            {"x0": 0.9, "x1": 0.9},
        ],
        output_data=[
            {"y": 1.0, "label": 0, "__turn": -1},
            {"y": 2.5, "label": 1, "__turn": 3},
            {"y": 4.0, "label": 1, "__turn": 5},
        ],
    )


def test_objective_summary_empty_dataset():
    schema = _schema()
    text = objective_summary(_empty_data(), schema, turn_log=[])
    assert "No evaluations completed yet" in text
    assert "maximize 'y'" in text


def test_objective_summary_reports_best_and_inputs():
    schema = _schema()
    text = objective_summary(_populated_data(), schema, turn_log=[])
    assert "Best objective so far: 4" in text
    assert "x0=0.9" in text and "x1=0.9" in text


def test_objective_summary_improvement_flag():
    schema = _schema()
    data = _populated_data()
    turn_log = [
        TurnRecord(
            turn_id=5,
            timestamp="t",
            agent_name="implementer",
            rationale="ran latin(1)",
            experiment_ids=[data.index[-1]],
        ),
    ]
    text = objective_summary(data, schema, turn_log)
    assert "improved this turn: yes" in text


def test_coverage_summary_empty():
    schema = _schema()
    text = coverage_summary(_empty_data(), schema, turn_log=[])
    assert text == "No evaluations completed yet."


def test_coverage_summary_continuous_and_categorical():
    schema = _schema()
    text = coverage_summary(_populated_data(), schema, turn_log=[])
    assert "Inputs:" in text and "Outputs:" in text
    # Categorical label distribution should show counts.
    assert "label (categorical)" in text
    assert "0: 1" in text and "1: 2" in text


def test_last_5_rationales_empty():
    schema = _schema()
    assert last_5_rationales(_empty_data(), schema, turn_log=[]) == ""


def test_last_5_rationales_includes_recent_records():
    schema = _schema()
    records = [
        TurnRecord(
            turn_id=i,
            timestamp=f"t{i}",
            agent_name="strategizer" if i % 2 == 0 else "implementer",
            rationale=f"thought {i}",
        )
        for i in range(7)
    ]
    text = last_5_rationales(_empty_data(), schema, records)
    lines = text.splitlines()
    assert len(lines) == 5
    # Oldest first: turn_ids 2..6 should appear.
    assert lines[0].startswith("[turn 2 |")
    assert lines[-1].startswith("[turn 6 |")
