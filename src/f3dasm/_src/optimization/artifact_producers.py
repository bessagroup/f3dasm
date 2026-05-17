"""Artifact producers for the agentic ``AnalysisBase``.

Each producer is a pure function ``(data, schema, turn_log) -> str`` that
returns the text the orchestrator stores under a fixed key in
``AnalysisBase``. The three Level-1 producers are:

- ``objective_summary``  — best-so-far, recent objectives, improvement flag
- ``coverage_summary``   — per-input and per-output coverage statistics
- ``last_5_rationales``  — rolling window of recent agent rationales

All producers gracefully handle the turn-0 case where the dataset has no
non-seed rows yet (``__turn = -1`` for seeds).
"""
#                                                                       Modules
# =============================================================================
from __future__ import annotations

from collections.abc import Iterable

# Standard
from statistics import mean, median

# Local
from ..experimentdata import ExperimentData
from .agent_dataclasses import ProblemSchema, TurnRecord

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Elvis Aguero (elvis_alexander_aguero_vera@brown.edu)"
__credits__ = ["Elvis Aguero"]
__status__ = "Experimental"
# =============================================================================


# Sentinel value stamped on rows that pre-existed before the loop began.
SEED_TURN = -1


def _objective_values_with_indices(
    data: ExperimentData, objective_name: str
) -> list[tuple[int, float]]:
    """Return the (row_index, objective) pairs with finite objective values.

    Parameters
    ----------
    data : ExperimentData
        The dataset to scan.
    objective_name : str
        Name of the output column holding the scalar objective.

    Returns
    -------
    list of tuple[int, float]
        ``(row_index, objective_value)`` pairs, excluding rows where the
        objective is missing, non-numeric, or NaN.
    """

    pairs: list[tuple[int, float]] = []
    for idx, sample in data.data.items():
        value = sample._output_data.get(objective_name)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if numeric != numeric:
            continue
        pairs.append((idx, numeric))
    return pairs


def _bounds_summary(schema: ProblemSchema) -> str:
    """Render the variable parameter bounds compactly.

    Parameters
    ----------
    schema : ProblemSchema
        Provides the ``variable_parameters`` mapping to render.

    Returns
    -------
    str
        Comma-separated bound descriptions, one per variable parameter.
        Returns ``"(no variable parameters)"`` when the mapping is empty.
    """

    parts: list[str] = []
    for name, spec in schema.variable_parameters.items():
        if spec.kind == "categorical" and spec.categories is not None:
            parts.append(f"{name} ∈ {spec.categories}")
        elif spec.bounds is not None:
            low, high = spec.bounds
            parts.append(f"{name} ∈ [{low:g}, {high:g}]")
        else:
            parts.append(f"{name} (unbounded)")
    return ", ".join(parts) if parts else "(no variable parameters)"


def objective_summary(
    data: ExperimentData,
    schema: ProblemSchema,
    turn_log: list[TurnRecord],
) -> str:
    """Produce the ``objective_summary`` artifact.

    Reports the best objective so far (with the row's input values), the
    last ten objectives in insertion order, a rolling mean, and whether
    the best improved during the most recent turn.

    Parameters
    ----------
    data : ExperimentData
        The shared dataset.
    schema : ProblemSchema
        Used to read ``objective_name`` and ``objective_direction``.
    turn_log : list of TurnRecord
        Used to determine the most recent turn's ``experiment_ids`` for
        the improvement flag.

    Returns
    -------
    str
        The artifact text.
    """

    pairs = _objective_values_with_indices(data, schema.objective_name)
    direction = schema.objective_direction
    if not pairs:
        return (
            "No evaluations completed yet. Search space: "
            f"{_bounds_summary(schema)}. Objective: "
            f"{direction} '{schema.objective_name}'."
        )

    if direction == "maximize":
        best_idx, best_val = max(pairs, key=lambda kv: kv[1])
    else:
        best_idx, best_val = min(pairs, key=lambda kv: kv[1])

    best_inputs = {
        name: data.data[best_idx]._input_data.get(name)
        for name in schema.variable_parameters
    }
    best_inputs_text = ", ".join(
        f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v!r}"
        for k, v in best_inputs.items()
    )

    last_objectives = [value for _, value in pairs[-10:]]
    rolling = mean(last_objectives) if last_objectives else float("nan")

    # Detect improvement during the most recent (non-error) turn.
    improved_this_turn = False
    last_turn = next(
        (record for record in reversed(turn_log) if record.experiment_ids),
        None,
    )
    if last_turn is not None:
        prior_pairs = [
            (idx, value)
            for idx, value in pairs
            if idx not in last_turn.experiment_ids
        ]
        if prior_pairs:
            if direction == "maximize":
                prior_best = max(value for _, value in prior_pairs)
                improved_this_turn = best_val > prior_best
            else:
                prior_best = min(value for _, value in prior_pairs)
                improved_this_turn = best_val < prior_best
        else:
            improved_this_turn = True

    lines = [
        f"Best objective so far: {best_val:.6g} at experiment_id={best_idx} "
        f"({best_inputs_text}).",
        f"Direction: {direction}. Total evaluated rows: {len(pairs)}.",
        f"Last {len(last_objectives)} objectives: "
        + ", ".join(f"{v:.4g}" for v in last_objectives)
        + f"; rolling mean = {rolling:.4g}.",
        "Best-so-far improved this turn: "
        + ("yes." if improved_this_turn else "no."),
    ]
    return "\n".join(lines)


def _numeric_values(rows: Iterable[dict], column: str) -> list[float]:
    """Collect finite numeric values for a given column across rows.

    Parameters
    ----------
    rows : iterable of dict
        Sequence of row dicts (input or output data dicts from
        ``ExperimentData``).
    column : str
        The column name whose values are to be collected.

    Returns
    -------
    list of float
        Finite (non-NaN) float values found under ``column`` in ``rows``.
        Rows missing the column or containing non-numeric values are skipped.
    """

    out: list[float] = []
    for row in rows:
        if column not in row:
            continue
        try:
            numeric = float(row[column])
        except (TypeError, ValueError):
            continue
        if numeric != numeric:
            continue
        out.append(numeric)
    return out


def coverage_summary(
    data: ExperimentData,
    schema: ProblemSchema,
    turn_log: list[TurnRecord],
) -> str:
    """Produce the ``coverage_summary`` artifact.

    For each variable parameter: count of evaluated rows, min/median/max,
    and the fraction of the bound range covered (max-min over bound
    width). For each declared output column: distribution (counts per
    value for categorical, min/median/max for continuous).

    Parameters
    ----------
    data : ExperimentData
        The shared dataset.
    schema : ProblemSchema
        Used to enumerate variable parameters and output columns.
    turn_log : list of TurnRecord
        Unused; kept for signature uniformity with the other producers.

    Returns
    -------
    str
        The artifact text.
    """

    input_rows = [sample._input_data for sample in data.data.values()]
    output_rows = [sample._output_data for sample in data.data.values()]

    if not input_rows:
        return "No evaluations completed yet."

    lines: list[str] = ["Inputs:"]
    for name, spec in schema.variable_parameters.items():
        values = _numeric_values(input_rows, name)
        if not values:
            lines.append(f"  {name}: no numeric values yet.")
            continue
        if spec.bounds is not None:
            low, high = spec.bounds
            span = high - low
            covered = (
                (max(values) - min(values)) / span if span > 0 else 0.0
            )
            covered_text = f"{covered:.2f}"
        else:
            covered_text = "n/a"
        lines.append(
            f"  {name}: n={len(values)}, min={min(values):.4g}, "
            f"median={median(values):.4g}, max={max(values):.4g}, "
            f"bound-fraction-covered={covered_text}."
        )

    lines.append("Outputs:")
    for name, spec in schema.output_columns.items():
        if spec.kind == "categorical":
            counts: dict[str, int] = {}
            for row in output_rows:
                value = row.get(name)
                if value is None:
                    continue
                counts[str(value)] = counts.get(str(value), 0) + 1
            if not counts:
                lines.append(f"  {name}: no values yet.")
            else:
                ordered = sorted(counts.items())
                rendered = ", ".join(f"{k}: {v}" for k, v in ordered)
                lines.append(f"  {name} (categorical): {{{rendered}}}.")
        else:
            values = _numeric_values(output_rows, name)
            if not values:
                lines.append(f"  {name}: no numeric values yet.")
            else:
                lines.append(
                    f"  {name}: n={len(values)}, min={min(values):.4g}, "
                    f"median={median(values):.4g}, max={max(values):.4g}."
                )

    return "\n".join(lines)


def last_5_rationales(
    data: ExperimentData,
    schema: ProblemSchema,
    turn_log: list[TurnRecord],
) -> str:
    """Produce the ``last_5_rationales`` artifact.

    Concatenates the rationale strings of the most recent five
    ``TurnRecord``s, oldest first, each prefixed by its ``turn_id`` and
    ``agent_name``.

    Returns
    -------
    str
        The artifact text. Empty string if no rationales are available
        (e.g. at turn 0).
    """

    recent = [record for record in turn_log if record.rationale][-5:]
    if not recent:
        return ""
    blocks: list[str] = []
    for record in recent:
        blocks.append(
            f"[turn {record.turn_id} | {record.agent_name}] {record.rationale}"
        )
    return "\n".join(blocks)


__all__ = [
    "objective_summary",
    "coverage_summary",
    "last_5_rationales",
    "SEED_TURN",
]
