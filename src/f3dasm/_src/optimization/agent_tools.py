"""Closure factories producing tool callables for the MVP agent tool sets.

Each factory returns a zero-argument or keyword-argument callable suitable
for passing directly to the Claude Agent SDK as a tool. The closures capture
immutable state (schema, specs, adapters) at construction time and therefore
never hold references to objects the orchestrator may rebind between turns.

The module is intentionally free of LLM imports — it is pure Python plumbing
that the orchestrator wires up. Tests can inject stubs for every dependency.

Default model (external reference only): ``claude-haiku-4-5-20251001``.
"""
#                                                                       Modules
# =============================================================================
from __future__ import annotations

# Standard
import dataclasses
import re
from collections.abc import Callable
from statistics import median
from typing import Any, Optional

# Local
from ..experimentdata import ExperimentData
from .agent_dataclasses import ProblemSchema, Strategy, StrategySpec
from .strategies import StrategyAdapter, validate_params

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Elvis Aguero (elvis_alexander_aguero_vera@brown.edu)"
__credits__ = ["Elvis Aguero"]
__status__ = "Experimental"
# =============================================================================


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_ALLOWED_OPS = {"==", "!=", "<", "<=", ">", ">="}

# One predicate: <identifier> <op> <literal>
# Group 1: column name, Group 2: operator, Group 3: value literal
_PRED_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(==|!=|<=|>=|<|>)\s*(.+)\s*$"
)


def _parse_filter(filter_str: str) -> Optional[tuple[str, str, Any]]:
    """Parse a single ``column op value`` predicate string.

    Parameters
    ----------
    filter_str : str
        The predicate to parse (e.g. ``"y == 1"``).

    Returns
    -------
    tuple of (str, str, object) or None
        ``(column, op, value)`` on success; ``None`` if the string is
        malformed or the operator is not in the allowed set.
    """

    match = _PRED_RE.match(filter_str)
    if match is None:
        return None
    column, op, raw_value = (
        match.group(1), match.group(2), match.group(3).strip()
    )
    if op not in _ALLOWED_OPS:
        return None
    # Reject value strings that start with a relational operator character:
    # this catches malformed inputs such as "y >> 1" where the regex
    # consumes the first ">" as the operator and leaves "> 1" as the value.
    if raw_value and raw_value[0] in "<>=!":
        return None

    # Try numeric coercion; fall back to string.
    try:
        value: Any = int(raw_value)
    except ValueError:
        try:
            value = float(raw_value)
        except ValueError:
            # Strip optional surrounding quotes.
            value = raw_value.strip("'\"")

    return column, op, value


def _apply_op(row_value: Any, op: str, threshold: Any) -> bool:
    """Evaluate ``row_value op threshold`` safely.

    Parameters
    ----------
    row_value : object
        The value extracted from the data row.
    op : str
        One of ``{==, !=, <, <=, >, >=}``.
    threshold : object
        The parsed predicate right-hand side.

    Returns
    -------
    bool
        Result of the comparison. Returns ``False`` for incompatible types
        or ``NaN`` operands, matching pandas default behaviour.
    """

    try:
        if op == "==":
            return row_value == threshold
        if op == "!=":
            return row_value != threshold
        # For ordered comparisons coerce both sides to float.
        lhs = float(row_value)
        rhs = float(threshold)
        # NaN comparisons always yield False.
        if lhs != lhs or rhs != rhs:
            return False
        if op == "<":
            return lhs < rhs
        if op == "<=":
            return lhs <= rhs
        if op == ">":
            return lhs > rhs
        if op == ">=":
            return lhs >= rhs
    except (TypeError, ValueError):
        return False
    return False


def _collect_numeric(values: list[Any]) -> list[float]:
    """Extract finite numeric values from a mixed list.

    Parameters
    ----------
    values : list of object
        Raw values from a data column.

    Returns
    -------
    list of float
        Finite (non-NaN) floats.
    """

    out: list[float] = []
    for v in values:
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f == f:
            out.append(f)
    return out


# ---------------------------------------------------------------------------
# Factory 1 — get_problem_schema
# ---------------------------------------------------------------------------


def make_get_problem_schema(schema: ProblemSchema) -> Callable[[], dict]:
    """Return a tool that serialises the problem schema to a dict.

    The closure captures a deep copy of ``schema`` so the agent cannot
    inadvertently mutate the live schema object.

    Parameters
    ----------
    schema : ProblemSchema
        The immutable problem description built by ``AgentOptimizer.arm``.

    Returns
    -------
    callable
        Zero-argument callable returning a ``dict`` representation of
        ``schema`` (produced via ``dataclasses.asdict``).
    """

    # Capture a deep copy so mutations of the returned dict never reach
    # the orchestrator's live schema.
    _schema_dict: dict = dataclasses.asdict(schema)

    def get_problem_schema() -> dict:
        """Return the problem schema as a deep-copied dict.

        Returns
        -------
        dict
            A complete, serialisable representation of ``ProblemSchema``.
            Modifying the returned dict has no effect on the live schema.
        """

        import copy
        return copy.deepcopy(_schema_dict)

    return get_problem_schema


# ---------------------------------------------------------------------------
# Factory 2 — list_strategies
# ---------------------------------------------------------------------------


def make_list_strategies(
    specs: dict[str, StrategySpec],
) -> Callable[[], list[dict]]:
    """Return a tool that lists every registered strategy as a dict.

    Parameters
    ----------
    specs : dict[str, StrategySpec]
        The spec map from ``default_registry()`` (or an augmented registry).

    Returns
    -------
    callable
        Zero-argument callable returning a list of dicts, one per strategy,
        each with keys ``name``, ``description``, and ``parameters``.
    """

    _specs_list: list[dict] = [
        dataclasses.asdict(spec) for spec in specs.values()
    ]

    def list_strategies() -> list[dict]:
        """Return the strategy registry entries as a list of dicts.

        Returns
        -------
        list of dict
            Each dict has keys ``name``, ``description``, and ``parameters``.
        """

        import copy
        return copy.deepcopy(_specs_list)

    return list_strategies


# ---------------------------------------------------------------------------
# Factory 3 — summarize_experiment_data
# ---------------------------------------------------------------------------


def make_summarize_experiment_data(
    data: ExperimentData,
    schema: ProblemSchema,
) -> Callable[[str, int], str]:
    """Return a tool that summarises the current ``ExperimentData``.

    Parameters
    ----------
    data : ExperimentData
        The live shared dataset (read by reference each call, not at
        factory time, so the tool always reflects the latest state).
    schema : ProblemSchema
        Provides the allowed filter columns and objective name.

    Returns
    -------
    callable
        Callable with signature ``(filter: str = "", top_k: int = 10) -> str``.
        On validation failure returns an ``"ERROR: ..."`` string that the
        agent reads in its tool result without raising an exception.
    """

    _valid_columns: frozenset[str] = frozenset(
        list(schema.variable_parameters) + list(schema.output_columns)
    )
    _objective: str = schema.objective_name
    _input_cols: list[str] = list(schema.variable_parameters)
    _output_cols: list[str] = list(schema.output_columns)

    def summarize_experiment_data(filter: str = "", top_k: int = 10) -> str:  # noqa: A002
        """Summarise the current dataset, optionally filtered by a predicate.

        Parameters
        ----------
        filter : str, default ''
            A single predicate of the form ``"column op value"`` where
            ``op`` is one of ``{==, !=, <, <=, >, >=}``. Empty string means
            no filter. Unknown columns and malformed predicates return an
            ``"ERROR: ..."`` string.
        top_k : int, default 10
            Number of top rows by objective to include in the output.

        Returns
        -------
        str
            Multi-line text summary, or ``"ERROR: ..."`` on bad input.
        """

        # -- Parse the filter predicate. ------------------------------------
        pred: Optional[tuple[str, str, Any]] = None
        if filter.strip():
            pred = _parse_filter(filter.strip())
            if pred is None:
                return (
                    f"ERROR: malformed filter predicate '{filter}'. "
                    "Expected: 'column op value' where op is one of "
                    f"{sorted(_ALLOWED_OPS)}."
                )
            col, op, threshold = pred
            if col not in _valid_columns:
                return (
                    f"ERROR: unknown column '{col}' in filter. "
                    f"Allowed columns: {sorted(_valid_columns)}."
                )

        # -- Collect all rows, applying the filter. -------------------------
        input_rows: list[dict] = []
        output_rows: list[dict] = []
        obj_with_inputs: list[tuple[float, dict]] = []

        for sample in data.data.values():
            inp = sample._input_data
            out = sample._output_data
            # Apply filter if present.
            if pred is not None:
                col, op, threshold = pred
                # The predicate column may be in inputs or outputs.
                row_value = inp.get(col, out.get(col))
                if not _apply_op(row_value, op, threshold):
                    continue

            input_rows.append(inp)
            output_rows.append(out)

            obj_val = out.get(_objective)
            if obj_val is not None:
                try:
                    obj_float = float(obj_val)
                    if obj_float == obj_float:
                        obj_with_inputs.append((obj_float, dict(inp)))
                except (TypeError, ValueError):
                    pass

        row_count = len(input_rows)
        lines: list[str] = [f"Rows matching filter: {row_count}"]

        # -- Per-input statistics. ------------------------------------------
        lines.append("\nInput variable statistics:")
        for col in _input_cols:
            vals = _collect_numeric(
                [row.get(col) for row in input_rows if col in row]
            )
            if not vals:
                lines.append(f"  {col}: no data")
                continue
            lines.append(
                f"  {col}: count={len(vals)}, "
                f"min={min(vals):.4g}, "
                f"median={median(vals):.4g}, "
                f"max={max(vals):.4g}"
            )

        # -- Per-output distribution. ----------------------------------------
        lines.append("\nOutput column distribution:")
        for out_col in _output_cols:
            out_spec = schema.output_columns.get(out_col)
            raw_vals = [
                row.get(out_col) for row in output_rows if out_col in row
            ]
            if not raw_vals:
                lines.append(f"  {out_col}: no data")
                continue

            if out_spec is not None and out_spec.kind == "categorical":
                counts: dict[str, int] = {}
                for v in raw_vals:
                    key = str(v)
                    counts[key] = counts.get(key, 0) + 1
                ordered = sorted(counts.items())
                rendered = ", ".join(f"{k}: {v}" for k, v in ordered)
                lines.append(f"  {out_col} (categorical): {{{rendered}}}")
            else:
                num_vals = _collect_numeric(raw_vals)
                if not num_vals:
                    lines.append(f"  {out_col}: no numeric data")
                else:
                    lines.append(
                        f"  {out_col} (continuous): "
                        f"min={min(num_vals):.4g}, "
                        f"median={median(num_vals):.4g}, "
                        f"max={max(num_vals):.4g}"
                    )

        # -- Top-K by objective. --------------------------------------------
        if obj_with_inputs:
            # Sort descending by objective value; caller may want max, but
            # we show the top regardless of direction — the agent knows its
            # direction from the schema.
            sorted_pairs = sorted(
                obj_with_inputs, key=lambda t: t[0], reverse=True
            )
            k = min(top_k, len(sorted_pairs))
            lines.append(f"\nTop-{k} rows by objective ({_objective}):")
            for rank, (obj_val, inp) in enumerate(sorted_pairs[:k], start=1):
                inp_str = ", ".join(
                    f"{c}={inp[c]:g}"
                    if isinstance(inp.get(c), (int, float))
                    else f"{c}={inp.get(c)!r}"
                    for c in _input_cols
                    if c in inp
                )
                lines.append(
                    f"  #{rank}: {_objective}={obj_val:.4g},"
                    f" inputs=({inp_str})"
                )

        return "\n".join(lines)

    return summarize_experiment_data


# ---------------------------------------------------------------------------
# Factory 4 — read_artifact
# ---------------------------------------------------------------------------


def make_read_artifact(
    analysis_base: dict[str, str],
) -> Callable[[str], str]:
    """Return a tool that reads a named artifact from the ``AnalysisBase``.

    Parameters
    ----------
    analysis_base : dict[str, str]
        The orchestrator-owned flat dict of named text artifacts. The
        closure holds a reference (not a copy) so it always reflects the
        latest artifact text written after the previous turn.

    Returns
    -------
    callable
        Callable with signature ``(key: str) -> str``. On a missing key
        returns an ``"ERROR: ..."`` string containing the available keys.
    """

    def read_artifact(key: str) -> str:
        """Return the named artifact text from the analysis base.

        Parameters
        ----------
        key : str
            Artifact name (e.g. ``'objective_summary'``).

        Returns
        -------
        str
            The artifact text, or ``"ERROR: no artifact named '<key>'; "``
            ``"available keys: [...]"`` if the key is absent.
        """

        if key in analysis_base:
            return analysis_base[key]
        available = sorted(analysis_base)
        return (
            f"ERROR: no artifact named '{key}'; "
            f"available keys: {available}"
        )

    return read_artifact


# ---------------------------------------------------------------------------
# Factory 5 — emit_strategy  (Strategizer-only)
# ---------------------------------------------------------------------------


def make_emit_strategy(
    specs: dict[str, StrategySpec],
    slot: list,
) -> Callable[[dict], str]:
    """Return a tool that validates and records a ``Strategy`` payload.

    The Strategizer calls this tool exactly once at the end of its turn.
    The orchestrator reads ``slot[0]`` after the session ends to retrieve
    the emitted strategy.

    Parameters
    ----------
    specs : dict[str, StrategySpec]
        The strategy spec map used for name and parameter validation.
    slot : list
        A single-element mutable list (initially empty) that the
        orchestrator allocates per turn. On success the factory appends
        the constructed ``Strategy`` object so the orchestrator can read
        ``slot[0]``.

    Returns
    -------
    callable
        Callable with signature ``(strategy_dict: dict) -> str`` returning
        ``"OK: strategy emitted."`` or an ``"ERROR: ..."`` string.
    """

    def emit_strategy(strategy_dict: dict) -> str:
        """Validate and emit a Strategy payload.

        Parameters
        ----------
        strategy_dict : dict
            Must have keys ``name`` (str), ``n_steps`` (int), ``params``
            (dict), and ``intent`` (str).

        Returns
        -------
        str
            ``"OK: strategy emitted."`` on success, or an ``"ERROR: ..."``
            string on validation failure.
        """

        required_keys = {"name", "n_steps", "params", "intent"}
        missing = required_keys - set(strategy_dict)
        if missing:
            return (
                "ERROR: strategy dict missing required keys: "
                f"{sorted(missing)}."
            )

        name = strategy_dict["name"]
        n_steps = strategy_dict["n_steps"]
        params = strategy_dict["params"]
        intent = strategy_dict["intent"]

        if name not in specs:
            available = sorted(specs)
            return (
                f"ERROR: unknown strategy name '{name}'. "
                f"Available: {available}."
            )

        error = validate_params(specs[name], params)
        if error is not None:
            return f"ERROR: {error}"

        try:
            n_steps_int = int(n_steps)
        except (TypeError, ValueError):
            return f"ERROR: n_steps must be an integer, got {n_steps!r}."

        strategy = Strategy(
            name=name,
            n_steps=n_steps_int,
            params=dict(params),
            intent=str(intent),
        )
        slot.append(strategy)
        return "OK: strategy emitted."

    return emit_strategy


# ---------------------------------------------------------------------------
# Factory 6 — run_strategy  (Implementer-only)
# ---------------------------------------------------------------------------


def make_run_strategy(
    adapters: dict[str, StrategyAdapter],
    specs: dict[str, StrategySpec],
    data: ExperimentData,
    domain: Any,
    data_generator: Any,
    objective_name: str,
    turn_id: int,
    strategy_calls_accumulator: list,
) -> Callable[[str, int, dict], str]:
    """Return a tool that validates and dispatches a registered strategy.

    Parameters
    ----------
    adapters : dict[str, StrategyAdapter]
        The adapter map from ``default_registry()``.
    specs : dict[str, StrategySpec]
        The spec map from ``default_registry()``.
    data : ExperimentData
        The live shared dataset. Mutated in place by the adapter.
    domain : Domain
        The f3dasm ``Domain`` passed to the adapter as a keyword argument.
    data_generator : DataGenerator
        The evaluator passed to the adapter as a keyword argument.
    objective_name : str
        Name of the objective output column, forwarded to the adapter.
    turn_id : int
        Current turn index, captured in the closure so every row the
        adapter writes receives the correct ``__turn`` stamp.
    strategy_calls_accumulator : list
        Mutable list that records ``(name, n_steps, params)`` tuples for
        each successful dispatch. The orchestrator reads it after the
        session ends to populate ``TurnRecord.strategy_calls``.

    Returns
    -------
    callable
        Callable with signature
        ``(name: str, n_steps: int, params: dict) -> str`` returning the
        strategy's ``StrategyOutcome.summary`` on success or an
        ``"ERROR: ..."`` string on validation failure.
    """

    def run_strategy(name: str, n_steps: int, params: dict) -> str:
        """Execute a registered strategy and return its outcome summary.

        Parameters
        ----------
        name : str
            Strategy name matching a ``StrategySpec`` in the registry.
        n_steps : int
            Step budget forwarded to the adapter.
        params : dict
            Parameter dict validated against the matching ``StrategySpec``.

        Returns
        -------
        str
            The human-readable ``StrategyOutcome.summary`` on success, or
            an ``"ERROR: ..."`` string on validation failure.
        """

        if name not in specs:
            available = sorted(specs)
            return (
                f"ERROR: unknown strategy '{name}'. "
                f"Available: {available}."
            )

        error = validate_params(specs[name], params)
        if error is not None:
            return f"ERROR: {error}"

        try:
            n_steps_int = int(n_steps)
        except (TypeError, ValueError):
            return f"ERROR: n_steps must be an integer, got {n_steps!r}."

        outcome = adapters[name](
            n_steps=n_steps_int,
            params=params,
            domain=domain,
            data=data,
            data_generator=data_generator,
            objective_name=objective_name,
            turn_id=turn_id,
        )
        strategy_calls_accumulator.append((name, n_steps_int, dict(params)))
        return outcome.summary

    return run_strategy


# ---------------------------------------------------------------------------
# Factory 7 — ask_strategizer  (Implementer-only)
# ---------------------------------------------------------------------------


def make_ask_strategizer(
    dispatch_fn: Callable[[str], str],
    budget: list[int],
) -> Callable[[str], str]:
    """Return a tool that forwards a clarification question to the Strategizer.

    Parameters
    ----------
    dispatch_fn : callable
        A thin wrapper the orchestrator provides. Receives the question
        string and returns the Strategizer's response string. In tests this
        should be a stub.
    budget : list of int
        Single-element mutable list containing the remaining follow-up
        allowance (e.g. ``[2]``). Decremented before each dispatch. When it
        reaches zero the tool returns an error string without calling
        ``dispatch_fn``.

    Returns
    -------
    callable
        Callable with signature ``(question: str) -> str``.
    """

    def ask_strategizer(question: str) -> str:
        """Send a clarification question to the Strategizer.

        The call is refused once the budget (a mutable counter in
        ``budget[0]``) reaches zero.

        Parameters
        ----------
        question : str
            The natural-language clarification request.

        Returns
        -------
        str
            The Strategizer's response, or an ``"ERROR: ..."`` string when
            the budget is exhausted.
        """

        if budget[0] <= 0:
            return (
                "ERROR: max follow-ups exhausted; proceed with the original "
                "Strategy or report failure."
            )
        budget[0] -= 1
        return dispatch_fn(question)

    return ask_strategizer


__all__ = [
    "make_get_problem_schema",
    "make_list_strategies",
    "make_summarize_experiment_data",
    "make_read_artifact",
    "make_emit_strategy",
    "make_run_strategy",
    "make_ask_strategizer",
]
