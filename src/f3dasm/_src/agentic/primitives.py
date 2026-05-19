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

from .agent_runtime import AgenticRunError, Delegation, _parse_delegation
from .backends.base import ImplementerSession, StrategizerSession

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Elvis Aguero (elvis_alexander_aguero_vera@brown.edu)"
__credits__ = ["Elvis Aguero"]
__status__ = "Experimental"
# =============================================================================

__all__ = [
    "parallel",
    "retry",
    "rounds",
]

# Union type accepted by all three primitives.
_AnySession = ImplementerSession | StrategizerSession


# ---------------------------------------------------------------------------
# parallel
# ---------------------------------------------------------------------------


def parallel(
    agents: list[_AnySession],
    task_fn: Callable[[int], str],
) -> list[Delegation]:
    """Fan-out: dispatch ``task_fn(k)`` to each agent concurrently.

    Each agent receives the string returned by ``task_fn(k)`` and must
    reply with a ``## Report`` block.  Replies are parsed into
    :class:`~agent_runtime.Delegation` objects.  Failures (parse errors
    or exceptions) are captured as delegations with ``conclusions`` set
    to the error description; no result is silently dropped.

    Parameters
    ----------
    agents : list
        Agent sessions.  All must implement ``send(str) -> str``.
    task_fn : callable
        Called with the zero-based agent index ``k``; must return the
        formatted task message to send to agent ``k``.

    Returns
    -------
    list[Delegation]
        One :class:`~agent_runtime.Delegation` per agent, in the same
        order as ``agents``.  Delegations for failed calls have
        ``conclusions`` set to an error description and ``metadata``
        containing ``{"error": True}``.
    """
    n = len(agents)
    if n == 0:
        return []
    results: list[Delegation] = [
        Delegation(intent="", expected_report="") for _ in range(n)
    ]

    def _call(k: int) -> tuple[int, str | Exception]:
        try:
            return k, agents[k].send(task_fn(k))
        except Exception as exc:
            return k, exc

    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as executor:
        futures = [executor.submit(_call, k) for k in range(n)]
        for future in concurrent.futures.as_completed(futures):
            k, outcome = future.result()
            deleg = Delegation(intent="", expected_report="")
            if isinstance(outcome, Exception):
                deleg.conclusions = f"ERROR: {outcome}"
                deleg.metadata["error"] = True
            else:
                parsed = _parse_delegation(outcome, deleg)
                if parsed is None:
                    deleg.conclusions = (
                        "ERROR: agent did not return a ## Report block."
                    )
                    deleg.raw = outcome
                    deleg.metadata["error"] = True
            results[k] = deleg

    return results


# ---------------------------------------------------------------------------
# retry
# ---------------------------------------------------------------------------


def retry(
    agent: _AnySession,
    task_msg: str,
    *,
    is_success: Callable[[Delegation], bool],
    max_fails: int = 3,
) -> Delegation:
    """Persistence loop: retry until ``is_success`` or ``max_fails``.

    Sends ``task_msg`` to ``agent``, parses the reply into a
    :class:`~agent_runtime.Delegation`, and tests it with ``is_success``.
    Retries on failure up to ``max_fails`` times.  Each retry re-sends
    the same ``task_msg`` (callers that need an adaptive message should
    use a closure that captures state).

    Parameters
    ----------
    agent : session
        Any object with ``send(str) -> str``.
    task_msg : str
        The formatted task message to send.
    is_success : callable
        Predicate on the parsed :class:`~agent_runtime.Delegation`.
        Return ``True`` to accept, ``False`` to retry.
    max_fails : int
        Maximum number of consecutive failures before raising.

    Returns
    -------
    Delegation
        The first :class:`~agent_runtime.Delegation` for which
        ``is_success`` returns ``True``.

    Raises
    ------
    AgenticRunError
        If ``is_success`` returns ``False`` (or parsing fails) for
        ``max_fails`` consecutive attempts.
    """
    fails = 0
    while True:
        reply = agent.send(task_msg)
        deleg = Delegation(intent="", expected_report="")
        parsed = _parse_delegation(reply, deleg)
        if parsed is not None and is_success(parsed):
            return parsed
        fails += 1
        if fails >= max_fails:
            raise AgenticRunError(
                f"retry: exceeded max_fails={max_fails}. "
                f"Last reply (first 200 chars): {reply[:200]!r}"
            )


# ---------------------------------------------------------------------------
# rounds
# ---------------------------------------------------------------------------


def rounds(
    agent_a: _AnySession,
    agent_b: _AnySession,
    n: int,
    initial: str,
) -> str:
    """Fixed-N debate: alternate ``n`` turns between two agents.

    Each round consists of ``agent_a`` responding to the current message,
    then ``agent_b`` responding to ``agent_a``'s reply.  After ``n``
    complete rounds the final response from ``agent_b`` is returned.

    This matches the AgenticSciML Proposer ↔ Critic debate (``n=4``):
    rounds 1–2 are analysis-only, round 3 is synthesis, round 4 is the
    final verdict.  Phase annotations may be embedded in ``initial`` for
    phase-aware agents.

    Parameters
    ----------
    agent_a : session
        First agent (e.g. Proposer).  Must implement ``send(str) -> str``.
    agent_b : session
        Second agent (e.g. Critic).  Must implement ``send(str) -> str``.
    n : int
        Number of complete A→B exchange rounds.  Must be ≥ 1.
    initial : str
        Opening message sent to ``agent_a`` to start the debate.

    Returns
    -------
    str
        The raw text reply from ``agent_b`` after ``n`` complete rounds.

    Raises
    ------
    ValueError
        If ``n < 1``.
    """
    if n < 1:
        raise ValueError(f"rounds: n must be ≥ 1, got {n}.")
    current = initial
    for _ in range(n):
        a_reply = agent_a.send(current)
        current = agent_b.send(a_reply)
    return current
