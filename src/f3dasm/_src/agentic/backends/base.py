"""Protocol contracts and Backend bundle for agentic-f3dasm backends.

Every LLM backend implementation must supply a :class:`Backend` instance
that bundles the five callables the orchestrator needs:

* a Strategizer session factory
* an Implementer session factory
* a preflight check (raises :class:`~agent_runtime.AgenticRunError` if
  the backend is not ready)
* a human-readable name
* the default model identifier for that backend

:class:`AgentSession` is the unified protocol for all agent sessions.
The legacy names :class:`StrategizerSession` and :class:`ImplementerSession`
are kept as aliases for backward compatibility.
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Unified session protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class AgentSession(Protocol):
    """Minimal protocol for any agent session.

    All agent roles — Strategizer, Implementer, Debugger, or any custom
    role — must implement this single method.  The session is responsible
    for maintaining conversation history across calls.

    Methods
    -------
    send(message)
        Send a user message; return the assistant reply.
    """

    def send(self, message: str) -> str:
        """Send a user message; return the assistant reply."""
        ...


# Backward-compatible aliases — existing code that imports these names
# continues to work without modification.
StrategizerSession = AgentSession
ImplementerSession = AgentSession


# ---------------------------------------------------------------------------
# Agent base class
# ---------------------------------------------------------------------------


class Agent:
    """Base class for all agentic nodes in a Graph.

    Subclasses override class-level attributes (``system_prompt``,
    ``tools``, etc.) to configure behaviour.  Behavioural differences
    belong in class attributes, not constructor arguments.

    Parameters
    ----------
    model : str or None
        Model identifier.  ``None`` delegates to the backend default.
    """

    system_prompt: str = ""
    tools: frozenset[str] = frozenset()
    reset_on_checkpoint: bool = True
    description: str | None = None

    def __init__(self, model: str | None = None) -> None:
        self.model = model

    def forward(self) -> None:
        """ADAS hook — override for inspectable Python orchestration."""


# ---------------------------------------------------------------------------
# Graph primitives
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Edge:
    """A directed delegation edge between two named agents.

    Parameters
    ----------
    source : str
        Name of the agent that is allowed to call ``Delegate``.
    target : str
        Name of the agent that receives the delegated task.
    """

    source: str
    target: str


@dataclass
class Graph:
    """Agent graph passed to :class:`~agent_runtime.AgenticRun`.

    Declares which agents exist, which directed delegation edges connect
    them, and which agent starts the run.  Loops are permitted.

    Parameters
    ----------
    nodes : dict[str, Agent]
        Maps unique agent names to :class:`Agent` instances.
    edges : sequence of Edge
        Directed delegation edges.  An agent with no outgoing edges
        receives no ``Delegate`` tool.
    entry : str
        Name of the agent that receives the initial briefing.
        Defaults to ``"strategizer"``.

    Raises
    ------
    ValueError
        If any edge endpoint names an undeclared node, or if *entry* is
        not declared.
    """

    nodes: dict  # dict[str, Agent]
    edges: tuple = ()
    entry: str = "strategizer"

    def __post_init__(self) -> None:
        self.edges = tuple(self.edges)
        bad = [k for k, v in self.nodes.items() if not isinstance(v, Agent)]
        if bad:
            raise TypeError(
                f"Graph nodes values must be Agent instances; "
                f"got invalid values for keys: {sorted(bad)}"
            )
        names = set(self.nodes)
        for e in self.edges:
            if e.source not in names or e.target not in names:
                raise ValueError(
                    f"Edge {e!r} references undeclared node. "
                    f"Declared names: {sorted(names)}"
                )
        if self.entry not in names:
            raise ValueError(
                f"entry={self.entry!r} not in nodes (entry node undeclared). "
                f"Declared names: {sorted(names)}"
            )

    def outgoing(self, name: str) -> list[str]:
        """Return target names for all edges out of *name*."""
        return [e.target for e in self.edges if e.source == name]


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
