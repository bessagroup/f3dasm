"""Tests for Level 2 firing primitives: parallel, retry, debate."""

from __future__ import annotations

from datetime import timedelta

import pytest

from f3dasm._src.agentic.agent_runtime import AgenticRunError, Delegation, Task
from f3dasm._src.agentic.primitives import debate, parallel, retry

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


def _task(intent: str = "do work", expected_report: str = "") -> Task:
    return Task(intent=intent, expected_report=expected_report)


# ===========================================================================
# parallel
# ===========================================================================


class TestParallel:
    def test_returns_one_delegation_per_agent(self):
        agents = [
            StubSession([_VALID_REPORT]),
            StubSession([_VALID_REPORT]),
        ]
        results = parallel(agents, task_fn=lambda k: _task(f"task {k}"))
        assert len(results) == 2

    def test_each_agent_receives_correct_task(self):
        agents = [StubSession([_VALID_REPORT]), StubSession([_VALID_REPORT])]
        parallel(agents, task_fn=lambda k: _task(f"task for agent {k}"))
        assert "task for agent 0" in agents[0].received[0]
        assert "task for agent 1" in agents[1].received[0]

    def test_task_stored_on_delegation(self):
        agents = [StubSession([_VALID_REPORT])]
        results = parallel(agents, task_fn=lambda k: _task(f"intent {k}"))
        assert results[0].task.intent == "intent 0"

    def test_parsed_conclusions(self):
        agents = [StubSession([_VALID_REPORT])]
        results = parallel(agents, task_fn=lambda k: _task())
        assert "Completed successfully" in results[0].report.conclusions

    def test_parsed_numbers(self):
        agents = [StubSession([_VALID_REPORT])]
        results = parallel(agents, task_fn=lambda k: _task())
        assert results[0].report.numbers["score"] == pytest.approx(0.8)

    def test_exception_captured_as_error_delegation(self):
        agents = [RaisingSession()]
        results = parallel(agents, task_fn=lambda k: _task())
        assert len(results) == 1
        assert results[0].metadata.get("error") is True
        assert "ERROR" in results[0].report.conclusions

    def test_unparseable_reply_captured_as_error_delegation(self):
        agents = [StubSession(["no report here"])]
        results = parallel(agents, task_fn=lambda k: _task())
        assert results[0].metadata.get("error") is True

    def test_mixed_success_and_failure(self):
        agents = [
            StubSession([_VALID_REPORT]),
            RaisingSession(),
            StubSession([_VALID_REPORT]),
        ]
        results = parallel(agents, task_fn=lambda k: _task())
        assert results[0].metadata.get("error") is not True
        assert results[1].metadata.get("error") is True
        assert results[2].metadata.get("error") is not True

    def test_order_preserved(self):
        reports = [
            _VALID_REPORT.replace("score: 0.8", f"score: {i}.0")
            for i in range(5)
        ]
        agents = [StubSession([r]) for r in reports]
        results = parallel(agents, task_fn=lambda k: _task(f"t{k}"))
        for k, result in enumerate(results):
            assert result.report.numbers["score"] == pytest.approx(float(k))

    def test_empty_agents_returns_empty(self):
        results = parallel([], task_fn=lambda k: _task())
        assert results == []


# ===========================================================================
# retry
# ===========================================================================


class TestRetry:
    def test_succeeds_on_first_attempt(self):
        agent = StubSession([_VALID_REPORT])
        result = retry(
            agent,
            _task(),
            is_success=lambda d: d.report.numbers.get("score", 0) > 0.5,
            max_fails=3,
        )
        assert result.report.numbers["score"] == pytest.approx(0.8)

    def test_retries_on_failure_then_succeeds(self):
        fail_report = _VALID_REPORT.replace("score: 0.8", "score: 0.1")
        agent = StubSession([fail_report, fail_report, _VALID_REPORT])
        result = retry(
            agent,
            _task(),
            is_success=lambda d: d.report.numbers.get("score", 0) > 0.5,
            max_fails=5,
        )
        assert result.report.numbers["score"] == pytest.approx(0.8)
        assert len(agent.received) == 3

    def test_raises_after_max_fails(self):
        fail_report = _VALID_REPORT.replace("score: 0.8", "score: 0.1")
        agent = StubSession([fail_report] * 10)
        with pytest.raises(AgenticRunError, match="max_fails=3"):
            retry(
                agent,
                _task(),
                is_success=lambda d: d.report.numbers.get("score", 0) > 0.5,
                max_fails=3,
            )
        assert len(agent.received) == 3

    def test_parse_failure_counts_as_fail(self):
        agent = StubSession(["no report block"] * 5)
        with pytest.raises(AgenticRunError):
            retry(
                agent,
                _task(),
                is_success=lambda d: True,
                max_fails=2,
            )
        assert len(agent.received) == 2

    def test_is_success_predicate_controls_retry(self):
        agent = StubSession([_VALID_REPORT])
        result = retry(
            agent,
            _task(),
            is_success=lambda d: "Completed" in d.report.conclusions,
            max_fails=1,
        )
        assert "Completed" in result.report.conclusions

    def test_task_stored_on_result(self):
        agent = StubSession([_VALID_REPORT])
        t = _task(intent="specific intent")
        result = retry(agent, t, is_success=lambda d: True, max_fails=1)
        assert result.task.intent == "specific intent"


# ===========================================================================
# debate
# ===========================================================================


class TestDebate:
    def test_single_round_returns_two_strings(self):
        agent_a = StubSession(["a_reply"])
        agent_b = StubSession(["b_reply"])
        result = debate(agent_a, agent_b, n=1, initial="start")
        assert result == ["a_reply", "b_reply"]

    def test_final_response_is_last_element(self):
        agent_a = StubSession(["a_reply"])
        agent_b = StubSession(["b_reply"])
        result = debate(agent_a, agent_b, n=1, initial="start")
        assert result[-1] == "b_reply"

    def test_two_rounds_returns_four_strings(self):
        agent_a = StubSession(["a1", "a2"])
        agent_b = StubSession(["b1", "b2"])
        result = debate(agent_a, agent_b, n=2, initial="start")
        assert result == ["a1", "b1", "a2", "b2"]

    def test_transcript_length_is_2n(self):
        agent_a = StubSession([f"a{i}" for i in range(4)])
        agent_b = StubSession([f"b{i}" for i in range(4)])
        result = debate(agent_a, agent_b, n=4, initial="proposal")
        assert len(result) == 8

    def test_four_rounds_final_response(self):
        agent_a = StubSession([f"a{i}" for i in range(4)])
        agent_b = StubSession([f"b{i}" for i in range(4)])
        result = debate(agent_a, agent_b, n=4, initial="proposal")
        assert result[-1] == "b3"

    def test_message_chaining(self):
        agent_a = StubSession(["a1", "a2"])
        agent_b = StubSession(["b1", "b2"])
        debate(agent_a, agent_b, n=2, initial="start")
        assert agent_a.received == ["start", "b1"]
        assert agent_b.received == ["a1", "a2"]

    def test_initial_passed_to_agent_a(self):
        agent_a = StubSession(["a_reply"])
        agent_b = StubSession(["b_reply"])
        debate(agent_a, agent_b, n=1, initial="my initial message")
        assert agent_a.received[0] == "my initial message"

    def test_zero_rounds_raises(self):
        agent_a = StubSession([])
        agent_b = StubSession([])
        with pytest.raises(ValueError, match="n must be"):
            debate(agent_a, agent_b, n=0, initial="start")

    def test_a_and_b_can_be_same_type(self):
        a = StubSession(["hello from a"])
        b = StubSession(["hello from b"])
        result = debate(a, b, n=1, initial="go")
        assert result[-1] == "hello from b"
