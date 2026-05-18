"""Protocol contracts and Backend bundle for agentic-f3dasm backends.

Every LLM backend implementation must supply a :class:`Backend` instance
that bundles the five callables the orchestrator needs:

* a Strategizer session factory
* an Implementer session factory
* a preflight check (raises :class:`~agent_runtime.AgenticRunError` if
  the backend is not ready)
* a human-readable name
* the default model identifier for that backend

The two :class:`typing.Protocol` classes below define the minimal
interface the orchestrator expects from each session object.
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Session protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class StrategizerSession(Protocol):
    """Minimal protocol for a Strategizer session.

    The orchestrator calls :meth:`send` and receives the full text of
    the assistant reply.  The session is responsible for maintaining
    conversation history across calls.

    Methods
    -------
    send(message)
        Send a user message; return the assistant reply.
    """

    def send(self, message: str) -> str:
        """Send a user message; return the assistant reply."""
        ...


@runtime_checkable
class ImplementerSession(Protocol):
    """Minimal protocol for an Implementer session.

    Methods
    -------
    send(message)
        Send a user message; return the assistant reply.
    """

    def send(self, message: str) -> str:
        """Send a user message; return the assistant reply."""
        ...


# ---------------------------------------------------------------------------
# Backend bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Backend:
    """Immutable bundle that fully describes one LLM backend.

    Pass an instance of this class to
    :class:`~f3dasm._src.agentic.agent_runtime.AgenticRun` via the
    ``backend`` keyword argument to swap the LLM provider without
    touching any other code.

    Parameters
    ----------
    name : str
        Human-readable identifier (e.g. ``"claude"``).
    default_model : str
        Model string used when the caller does not supply ``model=``.
    strategizer_factory : callable
        Factory with signature
        ``(*, system_prompt, model, tool_closures) -> StrategizerSession``.
    implementer_factory : callable
        Factory with signature
        ``(*, system_prompt, model, study_dir) -> ImplementerSession``.
    preflight : callable
        Zero-argument callable that raises
        :class:`~f3dasm._src.agentic.agent_runtime.AgenticRunError`
        when the backend is not ready (e.g. CLI binary missing).
        Must be a no-op when the backend is available.

    Notes
    -----
    The dataclass is frozen so that backend objects can be used as
    dictionary keys or cached without risk of mutation.
    """

    name: str
    default_model: str
    strategizer_factory: Callable[..., StrategizerSession]
    implementer_factory: Callable[..., ImplementerSession]
    preflight: Callable[[], None]
