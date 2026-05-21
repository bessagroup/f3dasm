"""Composable firing primitives for Level 2 agentic-f3dasm topologies.

Three plain-Python utilities for orchestrators:

- :func:`parallel` — fan-out: dispatch a task function to K agent
  sessions concurrently and collect :class:`~agent_runtime.Delegation`
  objects.
- :func:`retry` — persistence loop: retry until a success predicate is
  satisfied or ``max_fails`` is reached.
- :func:`rounds` — fixed-N debate: alternate N turns between two agent
  sessions and return the final response.

All three work with any object that implements the
:class:`~backends.base.ImplementerSession` Protocol
(``send(str) -> str``), so they compose with any backend.
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

import concurrent.futures
from collections.abc import Callable
from typing import Any

from .agent_runtime import AgenticRunError, Delegation, Report, Task, _format_task, _parse_delegation
from .backends.base import ImplementerSession, StrategizerSession

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Elvis Aguero (elvis_alexander_aguero_vera@brown.edu)"
__credits__ = ["Elvis Aguero"]
__status__ = "Experimental"
# =============================================================================

__all__ = [
    "debate",
    "parallel",
    "retry",
]

# Union type accepted by all three primitives.
_AnySession = ImplementerSession | StrategizerSession


# ---------------------------------------------------------------------------
# parallel
# ---------------------------------------------------------------------------


def parallel(
    agents: list[_AnySession],
    task_fn: Callable[[int], Task],
) -> list[Delegation]:
    """Fan-out: dispatch ``task_fn(k)`` to each agent concurrently.

    Each agent receives the formatted message for ``task_fn(k)`` and must
    reply with a ``## Report`` block.  Replies are parsed into
    :class:`~agent_runtime.Delegation` envelopes.  Failures (parse errors
    or exceptions) are captured as delegations with ``metadata["error"]``
    set to ``True``; no result is silently dropped.

    Parameters
    ----------
    agents : list
        Agent sessions.  All must implement ``send(str) -> str``.
    task_fn : callable
        Called with the zero-based agent index ``k``; must return the
        :class:`~agent_runtime.Task` to send to agent ``k``.

    Returns
    -------
    list[Delegation]
        One :class:`~agent_runtime.Delegation` per agent, in the same
        order as ``agents``.  Delegations for failed calls have
        ``metadata["error"] = True`` and the error description in
        ``report.conclusions``.
    """
    n = len(agents)
    if n == 0:
        return []

    _stub_task = Task(intent="", expected_report="")
    results: list[Delegation] = [Delegation(task=_stub_task) for _ in range(n)]

    def _call(k: int) -> tuple[int, Task, str | Exception]:
        task = task_fn(k)
        try:
            return k, task, agents[k].send(_format_task(task))
        except Exception as exc:
            return k, task, exc

    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as executor:
        futures = [executor.submit(_call, k) for k in range(n)]
        for future in concurrent.futures.as_completed(futures):
            k, task, outcome = future.result()
            deleg = Delegation(task=task)
            if isinstance(outcome, Exception):
                deleg.report = Report(
                    actions_taken="",
                    files_touched=[],
                    conclusions=f"ERROR: {outcome}",
                    numbers={},
                    raw="",
                )
                deleg.metadata["error"] = True
            else:
                if _parse_delegation(outcome, deleg) is None:
                    deleg.report = Report(
                        actions_taken="",
                        files_touched=[],
                        conclusions="ERROR: agent did not return a ## Report block.",
                        numbers={},
                        raw=outcome,
                    )
                    deleg.metadata["error"] = True
            results[k] = deleg

    return results


# ---------------------------------------------------------------------------
# retry
# ---------------------------------------------------------------------------

_DEFAULT_CORRECTIVE = (
    "Attempt {attempt} failed. Review what went wrong and revise your approach."
)


def retry(
    agent: _AnySession,
    task: Task,
    *,
    is_success: Callable[[Delegation], bool],
    max_fails: int = 3,
    on_failure: Callable[[Delegation, int], str] | None = None,
) -> Delegation:
    """Persistence loop: retry until ``is_success`` or ``max_fails``.

    Sends the formatted *task* to *agent*, parses the reply into a
    :class:`~agent_runtime.Delegation`, and tests it with ``is_success``.
    Retries on failure, sending the corrective message from *on_failure*
    (or a brief built-in message when *on_failure* is ``None``).

    Parameters
    ----------
    agent : session
        Any object with ``send(str) -> str``.
    task : Task
        The task to send.
    is_success : callable
        Predicate on the parsed :class:`~agent_runtime.Delegation`.
    max_fails : int
        Maximum consecutive failures before raising.
    on_failure : callable or None
        Called as ``on_failure(delegation, attempt)`` on each failure
        (attempt is 1-based).  Its return value is sent to the agent as a
        corrective message.  When ``None``, a brief built-in message is used.

    Raises
    ------
    AgenticRunError
        After ``max_fails`` consecutive failures.
    """
    task_msg = _format_task(task)
    fails = 0
    while True:
        reply = agent.send(task_msg)
        deleg = Delegation(task=task)
        parsed = _parse_delegation(reply, deleg)
        if parsed is not None and is_success(parsed):
            return parsed
        fails += 1
        if fails >= max_fails:
            raise AgenticRunError(
                f"retry: exceeded max_fails={max_fails}. "
                f"Last reply (first 200 chars): {reply[:200]!r}"
            )
        corrective = (
            on_failure(deleg, fails)
            if on_failure is not None
            else _DEFAULT_CORRECTIVE.format(attempt=fails)
        )
        task_msg = corrective


# ---------------------------------------------------------------------------
# rounds
# ---------------------------------------------------------------------------


def debate(
    agent_a: _AnySession,
    agent_b: _AnySession,
    n: int,
    initial: str,
) -> list[Delegation]:
    """Fixed-N debate: alternate ``n`` turns between two agents.

    Returns the full transcript as a list of :class:`~agent_runtime.Delegation`
    objects ``[a1, b1, a2, b2, …]`` with ``2*n`` elements.  Each Delegation's
    ``report.raw`` holds the agent's raw response.  The final response is
    ``debate(…)[-1].report.raw``.

    Unlike :func:`parallel` and :func:`retry`, debate does not require agents
    to emit a ``## Report`` block — responses are accepted as-is and stored in
    ``report.raw`` / ``report.conclusions``.

    Parameters
    ----------
    agent_a : session
        First agent (e.g. Proposer).
    agent_b : session
        Second agent (e.g. Critic).
    n : int
        Number of complete A→B exchange rounds.  Must be ≥ 1.
    initial : str
        Opening message sent to ``agent_a`` to start the debate.

    Returns
    -------
    list[Delegation]
        Interleaved transcript ``[a1, b1, a2, b2, …]`` of length ``2*n``.

    Raises
    ------
    ValueError
        If ``n < 1``.
    """
    if n < 1:
        raise ValueError(f"debate: n must be ≥ 1, got {n}.")

    def _wrap(raw: str, prompt: str) -> Delegation:
        task = Task(intent=prompt, expected_report="Respond to the debate.")
        d = Delegation(task=task)
        d.report = Report(
            actions_taken="",
            files_touched=[],
            conclusions=raw,
            numbers={},
            raw=raw,
        )
        return d

    transcript: list[Delegation] = []
    current = initial
    for _ in range(n):
        a_raw = agent_a.send(current)
        transcript.append(_wrap(a_raw, current))
        b_raw = agent_b.send(a_raw)
        transcript.append(_wrap(b_raw, a_raw))
        current = b_raw
    return transcript
