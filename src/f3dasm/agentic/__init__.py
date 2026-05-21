"""Public API for the agentic-f3dasm layer.

The architecture is a thin Python runtime over N persistent agent
sessions:

- **Strategizer** — reads any file, writes ``.md`` notes only,
  delegates execution to the Implementer.
- **Implementer** — reads/writes/executes freely inside the study tree,
  enriches and returns structured :class:`Delegation` envelopes.

The user's only required input is ``<study-dir>/PROBLEM_STATEMENT.md``.

Core symbols:

- ``AgenticRun`` — runtime entry point.
- ``AgenticOptimizer`` — f3dasm Optimizer interface wrapping AgenticRun.
- ``Task`` — request half of a round-trip exchange.
- ``Report`` — response half of a round-trip exchange.
- ``Delegation`` — envelope wrapping ``task`` + ``report`` + ``metadata``.
- ``AgenticRunError`` — raised for non-recoverable orchestrator errors.
- ``StudyConfig`` — per-study config loaded from ``config.yaml``.
- ``Backend`` — frozen dataclass bundling an LLM backend's factories.
- ``CLAUDE_BACKEND`` — the default Claude Agent SDK backend.
- ``LookupDataGenerator`` — nearest-neighbour pool evaluator.
- ``register_backend`` — register a custom backend by name.

Typed stores:

- ``AnalysisBase`` / ``AnalysisNode`` / ``ContextSlice`` — hierarchical
  per-solution analysis store.
- ``TaskRegistry`` / ``TaskStats`` — capped operator registry with
  success-rate tracking.

Firing primitives:

- ``parallel`` — fan-out to K agents concurrently.
- ``retry`` — persistence loop with configurable success predicate.
- ``debate`` — fixed-N raw-text exchange between two agents.
"""

#                                                                       Modules
# =============================================================================
from __future__ import annotations

from .._src.agentic.agent_runtime import (
    AgenticRun,
    AgenticRunError,
    Delegation,
    Report,
    StudyConfig,
    Task,
    register_backend,
)
from .._src.agentic.backends.base import Agent, Backend, Edge, Graph
from .._src.agentic.backends.claude import CLAUDE_BACKEND
from .._src.agentic.lookup import LookupDataGenerator
from .._src.agentic.optimizer import AgenticOptimizer
from .._src.agentic.primitives import debate, parallel, retry
from .._src.agentic.stores import (
    AnalysisBase,
    AnalysisNode,
    ContextSlice,
    TaskRegistry,
    TaskStats,
)

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Elvis Aguero (elvis_alexander_aguero_vera@brown.edu)"
__credits__ = ["Elvis Aguero"]
__status__ = "Experimental"
# =============================================================================


__all__ = [
    # Core
    "Agent",
    "AgenticRun",
    "Edge",
    "Graph",
    "AgenticRunError",
    "AgenticOptimizer",
    "Backend",
    "CLAUDE_BACKEND",
    "Delegation",
    "LookupDataGenerator",
    "Report",
    "StudyConfig",
    "Task",
    "register_backend",
    # Typed stores
    "AnalysisBase",
    "AnalysisNode",
    "ContextSlice",
    "TaskRegistry",
    "TaskStats",
    # Firing primitives
    "debate",
    "parallel",
    "retry",
]
