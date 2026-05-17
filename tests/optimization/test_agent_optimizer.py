"""Tests for AgentOptimizer, StrategizerImplementerOptimizer, and helpers.

This module verifies the turn-loop scaffolding, provenance recording,
artifact update, error handling, and narrate() logic of the agentic
optimization layer.  All tests use ``StubAgent`` in place of a live
Claude SDK session.
"""
#                                                                       Modules
# =============================================================================
from __future__ import annotations

# Standard
import json
from pathlib import Path
from typing import Any

# Third-party
import pytest

# Local
from f3dasm import ExperimentData
from f3dasm._src.datageneration.lookup import LookupDataGenerator
from f3dasm._src.optimization.agent_dataclasses import (
    OutputSpec,
    Strategy,
)
from f3dasm._src.optimization.agent_optimizer import (
    AgentExecutionError,
    AgentOptimizer,
    StrategizerImplementerOptimizer,
)
from f3dasm.design import Domain

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Elvis Aguero (elvis_alexander_aguero_vera@brown.edu)"
__credits__ = ["Elvis Aguero"]
__status__ = "Experimental"
# =============================================================================


# =============================================================================
# Helpers — 5-row pool matching M3 test substrate
# =============================================================================


def _build_pool_and_live() -> (
    tuple[ExperimentData, ExperimentData, LookupDataGenerator]
):
    """Build the standard 5-row 2D pool and a matching empty live dataset."""

    domain = Domain()
    domain.add_float("x0", low=0.0, high=1.0)
    domain.add_float("x1", low=0.0, high=1.0)
    domain.add_output("y")

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
    return pool, live, generator


def _build_live_with_seed() -> tuple[ExperimentData, LookupDataGenerator]:
    """Build a live dataset with one pre-existing seed row."""

    domain = Domain()
    domain.add_float("x0", low=0.0, high=1.0)
    domain.add_float("x1", low=0.0, high=1.0)
    domain.add_output("y")

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
    # Seed dataset: one row already present before arm().
    live = ExperimentData(
        domain=domain,
        input_data=[{"x0": 0.3, "x1": 0.3}],
        output_data=[{"y": 1.5}],
    )
    generator = LookupDataGenerator(pool, input_columns=["x0", "x1"])
    return live, generator


_OUTPUT_COLUMNS = {
    "y": OutputSpec(kind="continuous", description="test objective"),
}

_PHYSICS = "A 2D test function over [0,1]^2; objective is 'y'."


# =============================================================================
# StubAgent
# =============================================================================


class StubAgent:
    """Test double for an ``Agent`` that interacts with the tool list.

    The Strategizer stub finds ``emit_strategy`` in the tools list and
    calls it with a fixed ``Strategy``.  The Implementer stub finds
    ``run_strategy`` and calls it, optionally delegating to
    ``ask_strategizer`` first if ``ask_first`` is True.

    Parameters
    ----------
    name : str
        Agent name (``'strategizer'`` or ``'implementer'``).
    system_prompt : str
        Stored but not used by the stub.
    deny_list : sequence of str, optional
        Ignored by the stub.
    responses : list of str, optional
        Pre-canned response texts.  When exhausted a default rationale is
        returned.  Each entry must contain a ``## Rationale`` section.
    ask_first : bool, optional
        If True and the stub is an Implementer, it calls ``ask_strategizer``
        before ``run_strategy``.
    strategy_override : Strategy, optional
        Strategy payload emitted by a Strategizer stub (overrides default).
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        deny_list: Any = (),
        responses: Any = None,
        ask_first: bool = False,
        strategy_override: Any = None,
    ) -> None:
        """Initialise the stub.

        Parameters
        ----------
        name : str
            Agent name.
        system_prompt : str
            System prompt (unused by stub).
        deny_list : sequence of str, optional
            Ignored.
        responses : list of str, optional
            Pre-canned responses.
        ask_first : bool, optional
            Whether the Implementer calls ask_strategizer first.
        strategy_override : Strategy, optional
            Strategy to emit (Strategizer stub only).
        """

        self.name = name
        self.system_prompt = system_prompt
        self.responses = list(responses) if responses else []
        self.call_count = 0
        self.last_tools = None
        self._ask_first = ask_first
        self._strategy_override = strategy_override

    def send(self, message: str, tools: list) -> str:
        """Interact with the tools, then return a rationale response.

        Parameters
        ----------
        message : str
            Ignored by the stub.
        tools : list
            Tool callables exposed by the orchestrator.

        Returns
        -------
        str
            Response text containing ``## Rationale``.
        """

        self.last_tools = tools

        # Build a name->callable map for easy lookup.
        tool_map: dict[str, Any] = {}
        for tool in tools:
            tool_map[tool.__name__] = tool

        if self.name == "strategizer":
            # Strategizer: emit a strategy if emit_strategy is available.
            if "emit_strategy" in tool_map:
                strategy = self._strategy_override or Strategy(
                    name="latin",
                    n_steps=2,
                    params={},
                    intent="Explore the space with Latin sampling.",
                )
                tool_map["emit_strategy"](
                    {
                        "name": strategy.name,
                        "n_steps": strategy.n_steps,
                        "params": strategy.params,
                        "intent": strategy.intent,
                    }
                )
        else:
            # Implementer: optionally ask_strategizer, then run_strategy.
            if self._ask_first and "ask_strategizer" in tool_map:
                tool_map["ask_strategizer"]("What region should I focus on?")
            if "run_strategy" in tool_map:
                tool_map["run_strategy"]("latin", 2, {})

        # Return a pre-canned response or the default.
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
        else:
            response = "## Rationale\nStub default: completed turn."
        self.call_count += 1
        return response


class ErrorStubAgent(StubAgent):
    """Stub that always returns a response without a ## Rationale section."""

    def send(self, message: str, tools: list) -> str:
        """Return a response without a Rationale section.

        Parameters
        ----------
        message : str
            Ignored.
        tools : list
            Ignored.

        Returns
        -------
        str
            A plain string with no ``## Rationale`` section.
        """

        self.call_count += 1
        return "I have no rationale to offer."


# =============================================================================
# Factory helpers for tests
# =============================================================================


def _stub_strategizer_factory(**kwargs: Any) -> StubAgent:
    """Factory producing a Strategizer ``StubAgent``."""

    return StubAgent(name=kwargs.get("name", "strategizer"), **{
        k: v for k, v in kwargs.items() if k != "name"
    })


def _stub_implementer_factory(**kwargs: Any) -> StubAgent:
    """Factory producing an Implementer ``StubAgent``."""

    return StubAgent(name=kwargs.get("name", "implementer"), **{
        k: v for k, v in kwargs.items() if k != "name"
    })


def _error_strategizer_factory(**kwargs: Any) -> ErrorStubAgent:
    """Factory producing an always-error Strategizer stub."""

    return ErrorStubAgent(name=kwargs.get("name", "strategizer"), **{
        k: v for k, v in kwargs.items() if k != "name"
    })


def _error_implementer_factory(**kwargs: Any) -> ErrorStubAgent:
    """Factory producing an always-error Implementer stub."""

    return ErrorStubAgent(name=kwargs.get("name", "implementer"), **{
        k: v for k, v in kwargs.items() if k != "name"
    })


def _build_optimizer() -> StrategizerImplementerOptimizer:
    """Build a topology with normal stub agents."""

    return StrategizerImplementerOptimizer(
        strategizer_factory=_stub_strategizer_factory,
        implementer_factory=_stub_implementer_factory,
        physics_context=_PHYSICS,
        output_columns=_OUTPUT_COLUMNS,
        max_followups=2,
    )


def _build_error_optimizer() -> StrategizerImplementerOptimizer:
    """Build a topology with error-stub agents (no Rationale sections)."""

    return StrategizerImplementerOptimizer(
        strategizer_factory=_error_strategizer_factory,
        implementer_factory=_error_implementer_factory,
        physics_context=_PHYSICS,
        output_columns=_OUTPUT_COLUMNS,
        max_followups=2,
    )


# =============================================================================
# Tests — arm()
# =============================================================================


def test_arm_registers_turn_column() -> None:
    """arm() registers __turn in the domain output_space."""

    _, live, gen = _build_pool_and_live()
    opt = _build_optimizer()
    opt.arm(live, gen, "x", "y")
    assert "__turn" in live.domain.output_space


def test_arm_stamps_existing_rows_with_seed_turn() -> None:
    """arm() stamps all pre-existing rows with __turn = -1."""

    live, gen = _build_live_with_seed()
    opt = _build_optimizer()
    opt.arm(live, gen, "x", "y")

    for sample in live.data.values():
        assert sample._output_data.get("__turn") == -1


def test_arm_stamps_no_rows_when_live_is_empty() -> None:
    """arm() on an empty dataset stamps nothing (no rows to iterate)."""

    _, live, gen = _build_pool_and_live()
    opt = _build_optimizer()
    opt.arm(live, gen, "x", "y")
    # Empty dataset: no rows, nothing to assert on __turn values.
    assert len(live.data) == 0


def test_arm_initialises_analysis_base() -> None:
    """arm() populates all three AnalysisBase keys."""

    _, live, gen = _build_pool_and_live()
    opt = _build_optimizer()
    opt.arm(live, gen, "x", "y")
    assert "objective_summary" in opt.analysis_base
    assert "coverage_summary" in opt.analysis_base
    assert "last_5_rationales" in opt.analysis_base


def test_arm_builds_problem_schema() -> None:
    """arm() builds a ProblemSchema with variable_parameters from Domain."""

    _, live, gen = _build_pool_and_live()
    opt = _build_optimizer()
    opt.arm(live, gen, "x", "y")
    schema = opt.problem_schema
    assert "x0" in schema.variable_parameters
    assert "x1" in schema.variable_parameters
    assert schema.objective_name == "y"
    assert schema.objective_direction == "maximize"


# =============================================================================
# Tests — call() turn count and provenance
# =============================================================================


def test_call_2_iterations_produces_4_turn_records(tmp_path: Path) -> None:
    """2 iterations produce at least 4 TurnRecords (Strat + Impl x 2)."""

    _, live, gen = _build_pool_and_live()
    live._project_dir = tmp_path
    opt = _build_optimizer()
    opt.arm(live, gen, "x", "y")
    opt.call(live, n_iterations=2)

    assert len(opt.turn_log) >= 4


def test_call_turn_agent_names_pattern(tmp_path: Path) -> None:
    """TurnRecords alternate strategizer / implementer per iteration."""

    _, live, gen = _build_pool_and_live()
    live._project_dir = tmp_path
    opt = _build_optimizer()
    opt.arm(live, gen, "x", "y")
    opt.call(live, n_iterations=2)

    # First 4 records should follow Strat / Impl / Strat / Impl.
    names = [rec.agent_name for rec in opt.turn_log[:4]]
    expected = ["strategizer", "implementer", "strategizer", "implementer"]
    assert names == expected


def test_call_implementer_records_have_strategy_calls(tmp_path: Path) -> None:
    """Implementer TurnRecords carry at least one strategy_calls entry."""

    _, live, gen = _build_pool_and_live()
    live._project_dir = tmp_path
    opt = _build_optimizer()
    opt.arm(live, gen, "x", "y")
    opt.call(live, n_iterations=1)

    impl_records = [
        rec for rec in opt.turn_log if rec.agent_name == "implementer"
    ]
    assert impl_records, "No implementer records found."
    for rec in impl_records:
        assert len(rec.strategy_calls) >= 1


def test_call_strategizer_records_have_emitted_strategy(
    tmp_path: Path,
) -> None:
    """Strategizer TurnRecords carry an emitted_strategy payload."""

    _, live, gen = _build_pool_and_live()
    live._project_dir = tmp_path
    opt = _build_optimizer()
    opt.arm(live, gen, "x", "y")
    opt.call(live, n_iterations=1)

    strat_records = [
        rec for rec in opt.turn_log if rec.agent_name == "strategizer"
    ]
    assert strat_records, "No strategizer records found."
    assert strat_records[0].emitted_strategy is not None


# =============================================================================
# Tests — turn_log.jsonl persistence
# =============================================================================


def test_call_persists_turn_log_jsonl(tmp_path: Path) -> None:
    """After call(), turn_log.jsonl exists with the expected records."""

    _, live, gen = _build_pool_and_live()
    live._project_dir = tmp_path
    opt = _build_optimizer()
    opt.arm(live, gen, "x", "y")
    opt.call(live, n_iterations=1)

    log_path = tmp_path / "turn_log.jsonl"
    assert log_path.exists(), "turn_log.jsonl was not written."

    lines = [ln for ln in log_path.read_text().splitlines() if ln.strip()]
    assert len(lines) == len(opt.turn_log), (
        "Number of JSONL lines does not match turn_log length."
    )


def test_jsonl_content_matches_turn_log(tmp_path: Path) -> None:
    """Each JSONL line round-trips to the same turn_id as in turn_log."""

    _, live, gen = _build_pool_and_live()
    live._project_dir = tmp_path
    opt = _build_optimizer()
    opt.arm(live, gen, "x", "y")
    opt.call(live, n_iterations=1)

    log_path = tmp_path / "turn_log.jsonl"
    lines = [ln for ln in log_path.read_text().splitlines() if ln.strip()]
    for line, record in zip(lines, opt.turn_log, strict=True):
        parsed = json.loads(line)
        assert parsed["turn_id"] == record.turn_id
        assert parsed["agent_name"] == record.agent_name


# =============================================================================
# Tests — AgentExecutionError on consecutive errors
# =============================================================================


def test_three_consecutive_errors_raise_agent_execution_error(
    tmp_path: Path,
) -> None:
    """Three consecutive result='error' turns raise AgentExecutionError."""

    _, live, gen = _build_pool_and_live()
    live._project_dir = tmp_path
    opt = _build_error_optimizer()
    opt.arm(live, gen, "x", "y")

    with pytest.raises(AgentExecutionError):
        # Each iteration produces 2 error turns (strat + impl).
        # After 2 iterations we have 4 consecutive errors -> exceeds 3.
        opt.call(live, n_iterations=5)


# =============================================================================
# Tests — narrate()
# =============================================================================


def test_narrate_skips_seed_rows(tmp_path: Path) -> None:
    """narrate() produces output that does not mention __turn < 0 rows."""

    live, gen = _build_live_with_seed()
    live._project_dir = tmp_path
    opt = _build_optimizer()
    opt.arm(live, gen, "x", "y")

    # One iteration so narrate has something to work with.
    opt.call(live, n_iterations=1)
    text = opt.narrate()

    # The pre-existing seed row (with __turn = -1) should not appear in
    # narrate output; only rows from real turns should.
    # Verify that if any experiment_id appears it has __turn >= 0.
    for idx, sample in live.data.items():
        turn_stamp = sample._output_data.get("__turn")
        if turn_stamp is not None and int(turn_stamp) < 0:
            # Seed row: its index must not appear in the narration.
            assert f"experiment_id={idx}" not in text, (
                f"narrate() included seed row experiment_id={idx}."
            )


def test_narrate_returns_string(tmp_path: Path) -> None:
    """narrate() always returns a string (possibly empty)."""

    _, live, gen = _build_pool_and_live()
    live._project_dir = tmp_path
    opt = _build_optimizer()
    opt.arm(live, gen, "x", "y")
    opt.call(live, n_iterations=1)
    result = opt.narrate()
    assert isinstance(result, str)


# =============================================================================
# Tests — ask_strategizer follow-up
# =============================================================================


class AskFirstImplementerFactory:
    """Factory that builds an Implementer stub which calls ask_strategizer."""

    def __call__(
        self, name: str, system_prompt: str, **kwargs: Any
    ) -> StubAgent:
        """Build an ask-first Implementer stub.

        Parameters
        ----------
        name : str
            Agent name.
        system_prompt : str
            System prompt (unused).
        **kwargs : Any
            Extra arguments (ignored).

        Returns
        -------
        StubAgent
            A stub that calls ask_strategizer before run_strategy.
        """

        return StubAgent(
            name=name,
            system_prompt=system_prompt,
            ask_first=True,
        )


def test_followup_triggers_strategizer_subturn(tmp_path: Path) -> None:
    """Implementer calling ask_strategizer triggers a Strategizer sub-turn."""

    _, live, gen = _build_pool_and_live()
    live._project_dir = tmp_path

    ask_factory = AskFirstImplementerFactory()

    opt = StrategizerImplementerOptimizer(
        strategizer_factory=_stub_strategizer_factory,
        implementer_factory=ask_factory,
        physics_context=_PHYSICS,
        output_columns=_OUTPUT_COLUMNS,
        max_followups=2,
    )
    opt.arm(live, gen, "x", "y")
    opt.call(live, n_iterations=1)

    # The followup should have produced an additional strategizer turn.
    strat_records = [
        rec for rec in opt.turn_log if rec.agent_name == "strategizer"
    ]
    # At minimum: 1 main Strategizer turn + 1 followup Strategizer turn.
    assert len(strat_records) >= 2, (
        "Expected at least 2 strategizer turns when ask_strategizer is called."
    )


def test_followup_budget_exhaustion_returns_error_string(
    tmp_path: Path,
) -> None:
    """When max_followups is 0, ask_strategizer returns an error string."""

    _, live, gen = _build_pool_and_live()
    live._project_dir = tmp_path

    from f3dasm._src.optimization.agent_tools import make_ask_strategizer

    budget: list[int] = [0]
    ask_tool = make_ask_strategizer(
        dispatch_fn=lambda q: "OK",
        budget=budget,
    )
    result = ask_tool("Should I explore the corner?")
    assert "ERROR" in result
    assert "max follow-ups exhausted" in result


# =============================================================================
# Tests — render_context length guard
# =============================================================================


def test_render_context_truncates_long_rationales(tmp_path: Path) -> None:
    """_render_context warns and truncates when total > 8000 chars."""

    _, live, gen = _build_pool_and_live()
    live._project_dir = tmp_path
    opt = _build_optimizer()
    opt.arm(live, gen, "x", "y")

    # Artificially inflate the rationales artifact.
    opt.analysis_base["last_5_rationales"] = "X" * 9000
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ctx = opt._render_context()

    assert len(ctx) <= 8000 + 200, "Context was not meaningfully truncated."


# =============================================================================
# Tests — _extract_rationale
# =============================================================================


def test_extract_rationale_found() -> None:
    """_extract_rationale returns (text, 'ok') when ## Rationale is present."""

    response = "Some preamble.\n## Rationale\nThis worked well.\n\nEnd."
    rationale, result = AgentOptimizer._extract_rationale(response)
    assert result == "ok"
    assert "This worked well" in rationale


def test_extract_rationale_missing() -> None:
    """_extract_rationale returns ('', 'error') when section is absent."""

    response = "No heading here at all."
    rationale, result = AgentOptimizer._extract_rationale(response)
    assert result == "error"
    assert rationale == ""


def test_extract_rationale_stops_at_next_heading() -> None:
    """_extract_rationale stops extraction at the next ## heading."""

    response = "## Rationale\nGood.\n## Next Section\nIgnored."
    rationale, result = AgentOptimizer._extract_rationale(response)
    assert result == "ok"
    assert "Good" in rationale
    assert "Ignored" not in rationale
