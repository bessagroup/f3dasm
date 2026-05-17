"""Tests for the MVP tool closure factories in ``agent_tools``."""
#                                                                       Modules
# =============================================================================
from __future__ import annotations

# Standard
from unittest.mock import MagicMock

# Local
from f3dasm import ExperimentData
from f3dasm._src.datageneration.lookup import LookupDataGenerator
from f3dasm._src.optimization.agent_dataclasses import (
    OutputSpec,
    ParameterSpec,
    ProblemSchema,
)
from f3dasm._src.optimization.agent_tools import (
    make_ask_strategizer,
    make_emit_strategy,
    make_get_problem_schema,
    make_list_strategies,
    make_read_artifact,
    make_run_strategy,
    make_summarize_experiment_data,
)
from f3dasm._src.optimization.strategies import default_registry
from f3dasm.design import Domain

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Elvis Aguero (elvis_alexander_aguero_vera@brown.edu)"
__credits__ = ["Elvis Aguero"]
__status__ = "Experimental"
# =============================================================================


# =============================================================================
# Shared fixtures
# =============================================================================


def _schema() -> ProblemSchema:
    """Minimal two-input, two-output schema used across multiple tests."""
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
        physics_context="unit test context",
    )


def _build_data_and_domain() -> (
    tuple[ExperimentData, Domain, LookupDataGenerator]
):
    """Return a small pool, a live ExperimentData shell, and a generator."""
    domain = Domain()
    domain.add_float("x0", low=0.0, high=1.0)
    domain.add_float("x1", low=0.0, high=1.0)
    domain.add_output("y")
    domain.add_output("label")
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
            {"y": 1.0, "label": 0},
            {"y": 2.0, "label": 1},
            {"y": 5.0, "label": 1},
            {"y": 3.0, "label": 0},
            {"y": 4.0, "label": 1},
        ],
    )
    live = ExperimentData(domain=domain)
    generator = LookupDataGenerator(pool, input_columns=["x0", "x1"])
    return live, domain, generator


def _populated_data() -> ExperimentData:
    """Three-row dataset with known y and label values."""
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
            {"y": 2.5, "label": 1, "__turn": 0},
            {"y": 4.0, "label": 1, "__turn": 1},
        ],
    )


# =============================================================================
# Tests for make_get_problem_schema
# =============================================================================


def test_get_problem_schema_returns_expected_keys():
    """The dict must carry all top-level schema fields."""
    schema = _schema()
    tool = make_get_problem_schema(schema)
    result = tool()
    assert isinstance(result, dict)
    for key in (
        "variable_parameters",
        "forbidden_parameters",
        "objective_name",
        "objective_direction",
        "output_columns",
        "physics_context",
    ):
        assert key in result


def test_get_problem_schema_objective_name():
    """The returned dict must carry the correct objective name."""
    schema = _schema()
    tool = make_get_problem_schema(schema)
    assert tool()["objective_name"] == "y"


def test_get_problem_schema_is_not_live_reference():
    """Mutating the returned dict must not affect a subsequent call."""
    schema = _schema()
    tool = make_get_problem_schema(schema)
    first = tool()
    first["objective_name"] = "MUTATED"
    second = tool()
    assert second["objective_name"] == "y"


def test_get_problem_schema_variable_parameters_present():
    """variable_parameters should contain the two test parameters."""
    schema = _schema()
    tool = make_get_problem_schema(schema)
    vp = tool()["variable_parameters"]
    assert "x0" in vp and "x1" in vp


# =============================================================================
# Tests for make_list_strategies
# =============================================================================


def test_list_strategies_returns_one_dict_per_spec():
    """Each spec in the registry must appear as exactly one dict."""
    _, specs = default_registry()
    tool = make_list_strategies(specs)
    result = tool()
    assert isinstance(result, list)
    assert len(result) == len(specs)


def test_list_strategies_dict_has_required_keys():
    """Each entry must have name, description, and parameters keys."""
    _, specs = default_registry()
    tool = make_list_strategies(specs)
    for entry in tool():
        assert "name" in entry
        assert "description" in entry
        assert "parameters" in entry


def test_list_strategies_includes_latin():
    """The 'latin' strategy must appear in the list."""
    _, specs = default_registry()
    tool = make_list_strategies(specs)
    names = {d["name"] for d in tool()}
    assert "latin" in names


# =============================================================================
# Tests for make_summarize_experiment_data
# =============================================================================


def test_summarize_no_filter_includes_input_columns():
    """With no filter the summary must mention all input column names."""
    schema = _schema()
    data = _populated_data()
    tool = make_summarize_experiment_data(data, schema)
    result = tool()
    assert "x0" in result
    assert "x1" in result


def test_summarize_no_filter_mentions_row_count():
    """The summary must report how many rows matched."""
    schema = _schema()
    data = _populated_data()
    tool = make_summarize_experiment_data(data, schema)
    result = tool()
    assert "3" in result


def test_summarize_filter_y_eq_1_reduces_rows():
    """Filtering 'y == 1' on a known dataset should return fewer rows."""
    schema = _schema()
    data = _populated_data()
    tool = make_summarize_experiment_data(data, schema)
    tool()
    filtered = tool(filter="y == 1")
    # The filter matches one row (y=1.0), so the count must be smaller.
    assert "ERROR" not in filtered
    # Check that fewer rows are reported than the full summary.
    assert "Rows matching filter: 1" in filtered


def test_summarize_filter_unknown_column_returns_error():
    """An unknown column in the filter must produce an ERROR string."""
    schema = _schema()
    data = _populated_data()
    tool = make_summarize_experiment_data(data, schema)
    result = tool(filter="unknown_col == 1")
    assert result.startswith("ERROR")
    assert "unknown_col" in result


def test_summarize_filter_bad_operator_returns_error():
    """An unsupported operator (>>) must produce an ERROR string."""
    schema = _schema()
    data = _populated_data()
    tool = make_summarize_experiment_data(data, schema)
    result = tool(filter="y >> 1")
    assert result.startswith("ERROR")


def test_summarize_includes_top_k_rows():
    """The summary must include a top-K section when rows exist."""
    schema = _schema()
    data = _populated_data()
    tool = make_summarize_experiment_data(data, schema)
    result = tool(top_k=2)
    assert "Top-" in result


def test_summarize_categorical_output_shows_counts():
    """Categorical output columns must appear with their per-value counts."""
    schema = _schema()
    data = _populated_data()
    tool = make_summarize_experiment_data(data, schema)
    result = tool()
    assert "label (categorical)" in result


# =============================================================================
# Tests for make_read_artifact
# =============================================================================


def test_read_artifact_returns_value_for_existing_key():
    """A present key must return its string value."""
    base = {"objective_summary": "best so far: 42", "coverage_summary": "ok"}
    tool = make_read_artifact(base)
    assert tool("objective_summary") == "best so far: 42"


def test_read_artifact_error_on_missing_key():
    """A missing key must return an ERROR string listing available keys."""
    base = {"objective_summary": "x", "coverage_summary": "y"}
    tool = make_read_artifact(base)
    result = tool("nonexistent")
    assert result.startswith("ERROR")
    assert "nonexistent" in result
    assert "objective_summary" in result


def test_read_artifact_error_includes_all_available_keys():
    """The error message must enumerate every key in the base."""
    base = {"a": "1", "b": "2", "c": "3"}
    tool = make_read_artifact(base)
    result = tool("missing")
    for key in ("a", "b", "c"):
        assert key in result


# =============================================================================
# Tests for make_emit_strategy
# =============================================================================


def test_emit_strategy_valid_populates_slot():
    """A valid Strategy dict must return 'OK' and append to the slot."""
    _, specs = default_registry()
    slot: list = []
    tool = make_emit_strategy(specs, slot)
    result = tool(
        {
            "name": "latin",
            "n_steps": 5,
            "params": {"seed": 1},
            "intent": "explore",
        }
    )
    assert result == "OK: strategy emitted."
    assert len(slot) == 1
    assert slot[0].name == "latin"
    assert slot[0].n_steps == 5


def test_emit_strategy_unknown_name_returns_error():
    """An unregistered strategy name must produce an ERROR string."""
    _, specs = default_registry()
    slot: list = []
    tool = make_emit_strategy(specs, slot)
    result = tool(
        {"name": "nonexistent", "n_steps": 5, "params": {}, "intent": "x"}
    )
    assert result.startswith("ERROR")
    assert "nonexistent" in result


def test_emit_strategy_invalid_params_returns_error():
    """Invalid params (bad field) must produce an ERROR string."""
    _, specs = default_registry()
    slot: list = []
    tool = make_emit_strategy(specs, slot)
    result = tool(
        {
            "name": "latin",
            "n_steps": 5,
            "params": {"bad_field": 99},
            "intent": "explore",
        }
    )
    assert result.startswith("ERROR")
    assert len(slot) == 0


def test_emit_strategy_missing_required_fields_returns_error():
    """A dict missing required keys must produce an ERROR string."""
    _, specs = default_registry()
    slot: list = []
    tool = make_emit_strategy(specs, slot)
    result = tool({"name": "latin"})
    assert result.startswith("ERROR")


# =============================================================================
# Tests for make_run_strategy
# =============================================================================


def test_run_strategy_latin_dispatches_and_returns_summary():
    """Latin strategy must add rows and return a summary mentioning them."""
    live, domain, gen = _build_data_and_domain()
    adapters, specs = default_registry()
    accumulator: list = []

    tool = make_run_strategy(
        adapters=adapters,
        specs=specs,
        data=live,
        domain=domain,
        data_generator=gen,
        objective_name="y",
        turn_id=3,
        strategy_calls_accumulator=accumulator,
    )
    result = tool("latin", 4, {"seed": 7})
    assert "ERROR" not in result
    assert "Added 4 new rows" in result


def test_run_strategy_populates_accumulator():
    """Each successful dispatch must append an entry to the accumulator."""
    live, domain, gen = _build_data_and_domain()
    adapters, specs = default_registry()
    accumulator: list = []

    tool = make_run_strategy(
        adapters=adapters,
        specs=specs,
        data=live,
        domain=domain,
        data_generator=gen,
        objective_name="y",
        turn_id=0,
        strategy_calls_accumulator=accumulator,
    )
    tool("latin", 3, {})
    tool("random_uniform", 2, {"seed": 42})
    assert len(accumulator) == 2
    assert accumulator[0][0] == "latin"
    assert accumulator[1][0] == "random_uniform"


def test_run_strategy_unknown_name_returns_error():
    """An unregistered name must return an ERROR without touching the data."""
    live, domain, gen = _build_data_and_domain()
    adapters, specs = default_registry()
    accumulator: list = []

    tool = make_run_strategy(
        adapters=adapters,
        specs=specs,
        data=live,
        domain=domain,
        data_generator=gen,
        objective_name="y",
        turn_id=0,
        strategy_calls_accumulator=accumulator,
    )
    result = tool("fake_strategy", 5, {})
    assert result.startswith("ERROR")
    assert len(accumulator) == 0


def test_run_strategy_invalid_params_returns_error():
    """Unknown params must return an ERROR and not modify the accumulator."""
    live, domain, gen = _build_data_and_domain()
    adapters, specs = default_registry()
    accumulator: list = []

    tool = make_run_strategy(
        adapters=adapters,
        specs=specs,
        data=live,
        domain=domain,
        data_generator=gen,
        objective_name="y",
        turn_id=0,
        strategy_calls_accumulator=accumulator,
    )
    result = tool("latin", 3, {"unknown_param": 1})
    assert result.startswith("ERROR")
    assert len(accumulator) == 0


# =============================================================================
# Tests for make_ask_strategizer
# =============================================================================


def test_ask_strategizer_calls_dispatch_fn():
    """Each call within budget must forward the question to dispatch_fn."""
    dispatch = MagicMock(return_value="use sobol next")
    budget = [2]
    tool = make_ask_strategizer(dispatch, budget)
    result = tool("What region should I focus on?")
    assert result == "use sobol next"
    dispatch.assert_called_once_with("What region should I focus on?")


def test_ask_strategizer_decrements_budget():
    """Each call must decrement the budget counter."""
    dispatch = MagicMock(return_value="ok")
    budget = [2]
    tool = make_ask_strategizer(dispatch, budget)
    tool("q1")
    assert budget[0] == 1
    tool("q2")
    assert budget[0] == 0


def test_ask_strategizer_budget_two_allows_two_calls():
    """With budget=2 exactly two calls should succeed."""
    dispatch = MagicMock(return_value="response")
    budget = [2]
    tool = make_ask_strategizer(dispatch, budget)
    r1 = tool("first question")
    r2 = tool("second question")
    assert "ERROR" not in r1
    assert "ERROR" not in r2
    assert dispatch.call_count == 2


def test_ask_strategizer_third_call_returns_error():
    """The third call when budget=2 must return an ERROR string."""
    dispatch = MagicMock(return_value="response")
    budget = [2]
    tool = make_ask_strategizer(dispatch, budget)
    tool("q1")
    tool("q2")
    result = tool("q3")
    assert result.startswith("ERROR")
    # dispatch_fn must NOT be called a third time.
    assert dispatch.call_count == 2


def test_ask_strategizer_zero_budget_immediately_errors():
    """With budget=0 the first call must already return ERROR."""
    dispatch = MagicMock(return_value="should not be reached")
    budget = [0]
    tool = make_ask_strategizer(dispatch, budget)
    result = tool("anything")
    assert result.startswith("ERROR")
    dispatch.assert_not_called()
