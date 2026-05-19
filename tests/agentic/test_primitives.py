"""Tests for Level 2 firing primitives: parallel, retry, rounds."""

from __future__ import annotations

from datetime import timedelta

import pytest

from f3dasm._src.agentic.agent_runtime import AgenticRunError, Delegation
from f3dasm._src.agentic.primitives import parallel, retry, rounds

__author__ = "Elvis Aguero (elvis_alexander_aguero_vera@brown.edu)"
__credits__ = ["Elvis Aguero"]
__status__ = "Experimental"


# ---------------------------------------------------------------------------
# Stub sessions
# ---------------------------------------------------------------------------

_VALID_REPORT = """\
## Report

### Actions taken
- Ran the task.

### Files touched
- workspace/result.csv

### Conclusions
Completed successfully.

### Numbers
score: 0.8
"""


class StubSession:
    """Deterministic stub for ImplementerSession / StrategizerSession."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self._index = 0
        self.received: list[str] = []

    def send(self, message: str) -> str:
        self.received.append(message)
        if self._index >= len(self._replies):
            return ""
        reply = self._replies[self._index]
        self._index += 1
        return reply


class RaisingSession:
    """Session that raises on send()."""

    def send(self, message: str) -> str:
        raise RuntimeError("simulated backend failure")


# ===========================================================================
# parallel
# ===========================================================================


class TestParallel:
    def test_returns_one_delegation_per_agent(self):
        agents = [
            StubSession([_VALID_REPORT]),
            StubSession([_VALID_REPORT]),
        ]
        results = parallel(agents, task_fn=lambda k: f"task {k}")
        assert len(results) == 2

    def test_each_agent_receives_correct_task(self):
        agents = [StubSession([_VALID_REPORT]), StubSession([_VALID_REPORT])]
        parallel(agents, task_fn=lambda k: f"task for agent {k}")
        assert agents[0].received == ["task for agent 0"]
        assert agents[1].received == ["task for agent 1"]

    def test_parsed_conclusions(self):
        agents = [StubSession([_VALID_REPORT])]
        results = parallel(agents, task_fn=lambda k: "t")
        assert "Completed successfully" in results[0].conclusions

    def test_parsed_numbers(self):
        agents = [StubSession([_VALID_REPORT])]
        results = parallel(agents, task_fn=lambda k: "t")
        assert results[0].numbers["score"] == pytest.approx(0.8)

    def test_exception_captured_as_error_delegation(self):
        agents = [RaisingSession()]
        results = parallel(agents, task_fn=lambda k: "t")
        assert len(results) == 1
        assert results[0].metadata.get("error") is True
        assert "ERROR" in results[0].conclusions

    def test_unparseable_reply_captured_as_error_delegation(self):
        agents = [StubSession(["no report here"])]
        results = parallel(agents, task_fn=lambda k: "t")
        assert results[0].metadata.get("error") is True

    def test_mixed_success_and_failure(self):
        agents = [
            StubSession([_VALID_REPORT]),
            RaisingSession(),
            StubSession([_VALID_REPORT]),
        ]
        results = parallel(agents, task_fn=lambda k: "t")
        assert results[0].metadata.get("error") is not True
        assert results[1].metadata.get("error") is True
        assert results[2].metadata.get("error") is not True

    def test_order_preserved(self):
        reports = [
            _VALID_REPORT.replace("score: 0.8", f"score: {i}.0")
            for i in range(5)
        ]
        agents = [StubSession([r]) for r in reports]
        results = parallel(agents, task_fn=lambda k: f"t{k}")
        for k, result in enumerate(results):
            assert result.numbers["score"] == pytest.approx(float(k))

    def test_empty_agents_returns_empty(self):
        results = parallel([], task_fn=lambda k: "t")
        assert results == []


# ===========================================================================
# retry
# ===========================================================================


class TestRetry:
    def test_succeeds_on_first_attempt(self):
        agent = StubSession([_VALID_REPORT])
        result = retry(
            agent,
            "task",
            is_success=lambda d: d.numbers.get("score", 0) > 0.5,
            max_fails=3,
        )
        assert result.numbers["score"] == pytest.approx(0.8)

    def test_retries_on_failure_then_succeeds(self):
        fail_report = _VALID_REPORT.replace("score: 0.8", "score: 0.1")
        agent = StubSession([fail_report, fail_report, _VALID_REPORT])
        result = retry(
            agent,
            "task",
            is_success=lambda d: d.numbers.get("score", 0) > 0.5,
            max_fails=5,
        )
        assert result.numbers["score"] == pytest.approx(0.8)
        assert len(agent.received) == 3

    def test_raises_after_max_fails(self):
        fail_report = _VALID_REPORT.replace("score: 0.8", "score: 0.1")
        agent = StubSession([fail_report] * 10)
        with pytest.raises(AgenticRunError, match="max_fails=3"):
            retry(
                agent,
                "task",
                is_success=lambda d: d.numbers.get("score", 0) > 0.5,
                max_fails=3,
            )
        assert len(agent.received) == 3

    def test_parse_failure_counts_as_fail(self):
        agent = StubSession(["no report block"] * 5)
        with pytest.raises(AgenticRunError):
            retry(
                agent,
                "task",
                is_success=lambda d: True,
                max_fails=2,
            )
        assert len(agent.received) == 2

    def test_is_success_predicate_controls_retry(self):
        agent = StubSession([_VALID_REPORT])
        result = retry(
            agent,
            "task",
            is_success=lambda d: "Completed" in d.conclusions,
            max_fails=1,
        )
        assert "Completed" in result.conclusions


# ===========================================================================
# rounds
# ===========================================================================


class TestRounds:
    def test_single_round(self):
        agent_a = StubSession(["a_reply"])
        agent_b = StubSession(["b_reply"])
        result = rounds(agent_a, agent_b, n=1, initial="start")
        assert result == "b_reply"
        assert agent_a.received == ["start"]
        assert agent_b.received == ["a_reply"]

    def test_two_rounds(self):
        agent_a = StubSession(["a1", "a2"])
        agent_b = StubSession(["b1", "b2"])
        result = rounds(agent_a, agent_b, n=2, initial="start")
        assert result == "b2"
        assert agent_a.received == ["start", "b1"]
        assert agent_b.received == ["a1", "a2"]

    def test_four_rounds_agenticsciml_style(self):
        agent_a = StubSession([f"a{i}" for i in range(4)])
        agent_b = StubSession([f"b{i}" for i in range(4)])
        result = rounds(agent_a, agent_b, n=4, initial="proposal")
        assert result == "b3"
        assert len(agent_a.received) == 4
        assert len(agent_b.received) == 4

    def test_initial_passed_to_agent_a(self):
        agent_a = StubSession(["a_reply"])
        agent_b = StubSession(["b_reply"])
        rounds(agent_a, agent_b, n=1, initial="my initial message")
        assert agent_a.received[0] == "my initial message"

    def test_zero_rounds_raises(self):
        agent_a = StubSession([])
        agent_b = StubSession([])
        with pytest.raises(ValueError, match="n must be"):
            rounds(agent_a, agent_b, n=0, initial="start")

    def test_a_and_b_can_be_same_type(self):
        a = StubSession(["hello from a"])
        b = StubSession(["hello from b"])
        result = rounds(a, b, n=1, initial="go")
        assert result == "hello from b"
