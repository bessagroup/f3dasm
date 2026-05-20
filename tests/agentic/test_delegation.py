"""Tests for the Level 2 Delegation class and its format/parse helpers."""

from __future__ import annotations

from datetime import timedelta

import pytest

from f3dasm._src.agentic.agent_runtime import (
    Delegation,
    Task,
    _format_delegation,
    _parse_delegation,
)

__author__ = "Elvis Aguero (elvis_alexander_aguero_vera@brown.edu)"
__credits__ = ["Elvis Aguero"]
__status__ = "Experimental"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_REPORT = """\
## Report

### Actions taken
- Did the thing.

### Files touched
- /study/workspace/out.csv

### Conclusions
Task succeeded. No anomalies.

### Numbers
best_x: 0.5
n_evaluated: 10
"""


def _make_delegation(
    intent: str = "run LHS",
    expected_report: str = "best_x and n_evaluated",
    remaining_time: timedelta | None = None,
    budget: timedelta | None = None,
) -> Delegation:
    return Delegation(task=Task(
        intent=intent,
        expected_report=expected_report,
        remaining_time=remaining_time,
        budget=budget,
    ))


# ---------------------------------------------------------------------------
# Delegation construction
# ---------------------------------------------------------------------------


def test_delegation_defaults():
    d = _make_delegation()
    assert d.report is None
    assert d.is_complete is False
    assert d.metadata == {}


def test_delegation_is_complete_after_parse():
    d = _make_delegation()
    _parse_delegation(_VALID_REPORT, d)
    assert d.is_complete is True
    assert d.report is not None


def test_delegation_metadata_isolated():
    d1 = _make_delegation()
    d2 = _make_delegation()
    d1.metadata["key"] = "value"
    assert "key" not in d2.metadata


# ---------------------------------------------------------------------------
# _format_delegation
# ---------------------------------------------------------------------------


def test_format_delegation_contains_intent():
    d = _make_delegation(intent="run LHS sampling")
    msg = _format_delegation(d)
    assert "run LHS sampling" in msg


def test_format_delegation_contains_expected_report():
    d = _make_delegation(expected_report="report the best_x value")
    msg = _format_delegation(d)
    assert "report the best_x value" in msg


def test_format_delegation_has_task_heading():
    d = _make_delegation()
    msg = _format_delegation(d)
    assert "## Task" in msg


def test_format_delegation_no_warning_without_budget():
    d = _make_delegation(remaining_time=timedelta(minutes=5))
    msg = _format_delegation(d)
    assert "Budget nearly exhausted" not in msg


def test_format_delegation_no_warning_ample_budget():
    d = _make_delegation(
        remaining_time=timedelta(minutes=50),
        budget=timedelta(hours=1),
    )
    msg = _format_delegation(d)
    assert "Budget nearly exhausted" not in msg


def test_format_delegation_warning_at_low_budget():
    d = _make_delegation(
        remaining_time=timedelta(minutes=5),
        budget=timedelta(hours=1),
    )
    msg = _format_delegation(d)
    assert "Budget nearly exhausted" in msg


def test_format_delegation_shows_remaining_time():
    d = _make_delegation(
        remaining_time=timedelta(hours=1, minutes=30),
        budget=timedelta(hours=2),
    )
    msg = _format_delegation(d)
    assert "01:30:00" in msg


# ---------------------------------------------------------------------------
# _parse_delegation
# ---------------------------------------------------------------------------


def test_parse_delegation_fills_actions():
    d = _make_delegation()
    result = _parse_delegation(_VALID_REPORT, d)
    assert result is not None
    assert "Did the thing" in result.report.actions_taken


def test_parse_delegation_fills_files_touched():
    d = _make_delegation()
    result = _parse_delegation(_VALID_REPORT, d)
    assert result is not None
    assert "/study/workspace/out.csv" in result.report.files_touched


def test_parse_delegation_fills_conclusions():
    d = _make_delegation()
    result = _parse_delegation(_VALID_REPORT, d)
    assert result is not None
    assert "Task succeeded" in result.report.conclusions


def test_parse_delegation_fills_numbers():
    d = _make_delegation()
    result = _parse_delegation(_VALID_REPORT, d)
    assert result is not None
    assert result.report.numbers["best_x"] == pytest.approx(0.5)
    assert result.report.numbers["n_evaluated"] == pytest.approx(10.0)


def test_parse_delegation_fills_raw():
    d = _make_delegation()
    result = _parse_delegation(_VALID_REPORT, d)
    assert result is not None
    assert "## Report" in result.report.raw


def test_parse_delegation_preserves_intent():
    d = _make_delegation(intent="my original intent")
    result = _parse_delegation(_VALID_REPORT, d)
    assert result is not None
    assert result.task.intent == "my original intent"


def test_parse_delegation_preserves_metadata():
    d = _make_delegation()
    d.metadata["channel"] = "strategizer->implementer"
    result = _parse_delegation(_VALID_REPORT, d)
    assert result is not None
    assert result.metadata["channel"] == "strategizer->implementer"


def test_parse_delegation_returns_none_on_missing_heading():
    d = _make_delegation()
    result = _parse_delegation("No report here.", d)
    assert result is None


def test_parse_delegation_returns_same_object():
    d = _make_delegation()
    result = _parse_delegation(_VALID_REPORT, d)
    assert result is d


# ---------------------------------------------------------------------------
# Round-trip: format then parse
# ---------------------------------------------------------------------------


def test_round_trip_format_parse():
    d = _make_delegation(intent="run SMAC optimisation", expected_report="best_y")
    formatted = _format_delegation(d)
    assert "run SMAC optimisation" in formatted
    assert "best_y" in formatted

    reply = formatted + "\n" + _VALID_REPORT
    filled = _parse_delegation(reply, d)
    assert filled is not None
    assert filled.task.intent == "run SMAC optimisation"
    assert filled.report.numbers["best_x"] == pytest.approx(0.5)
