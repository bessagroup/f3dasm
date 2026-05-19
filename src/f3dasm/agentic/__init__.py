"""Public API for the agentic-f3dasm layer.

The architecture is a thin Python runtime over N persistent agent
sessions:

- **Strategizer** — reads any file, writes ``.md`` notes only,
  delegates execution to the Implementer.
- **Implementer** — reads/writes/executes freely inside the study tree,
  enriches and returns structured :class:`Delegation` objects.

The user's only required input is ``<study-dir>/PROBLEM_STATEMENT.md``.

Level-1 public symbols (stable):

- ``AgenticRun`` — runtime entry point.
- ``Task`` — backward-compatible alias; use :class:`Delegation` for new code.
- ``Report`` — backward-compatible alias; use :class:`Delegation` for new code.
- ``Delegation`` — symmetric exchange class; both parties enrich the same object.
- ``AgenticRunError`` — raised for non-recoverable orchestrator errors.
- ``CHECKPOINT_EVERY`` — Implementer-call cadence.
- ``Backend`` — frozen dataclass bundling an LLM backend's factories.
- ``CLAUDE_BACKEND`` — the default Claude Agent SDK backend.
- ``OLLAMA_BACKEND`` — Ollama backend (local models).
- ``MVP_DEFAULT_MODEL`` — default model id.
- ``LookupDataGenerator`` — nearest-neighbour pool evaluator.

Level-2 public symbols (typed stores + firing primitives):

- ``AnalysisBase`` / ``AnalysisNode`` / ``AnalysisSlice`` — hierarchical
  per-solution analysis store.
- ``TaskRegistry`` / ``RegistryEntry`` / ``MAX_TASKS`` — capped operator
  registry with success-rate tracking.
- ``parallel`` — fan-out to K agents concurrently.
- ``retry`` — persistence loop with configurable success predicate.
- ``rounds`` — fixed-N debate between two agents.
"""

#                                                                       Modules
# =============================================================================
from __future__ import annotations

from .._src.agentic.agent_runtime import (
    CHECKPOINT_EVERY,
    MVP_DEFAULT_MODEL,
    AgenticRun,
    AgenticRunError,
    Delegation,
    Report,
    StudyConfig,
    Task,
)
from .._src.agentic.backends.base import Backend
from .._src.agentic.backends.claude import CLAUDE_BACKEND
from .._src.agentic.backends.ollama import OLLAMA_BACKEND
from .._src.agentic.lookup import LookupDataGenerator
from .._src.agentic.primitives import parallel, retry, rounds
from .._src.agentic.stores import (
    MAX_TASKS,
    AnalysisBase,
    AnalysisNode,
    AnalysisSlice,
    RegistryEntry,
    TaskRegistry,
)

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Elvis Aguero (elvis_alexander_aguero_vera@brown.edu)"
__credits__ = ["Elvis Aguero"]
__status__ = "Experimental"
# =============================================================================


__all__ = [
    # Level 1
    "AgenticRun",
    "AgenticRunError",
    "Backend",
    "CHECKPOINT_EVERY",
    "CLAUDE_BACKEND",
    "Delegation",
    "LookupDataGenerator",
    "MVP_DEFAULT_MODEL",
    "OLLAMA_BACKEND",
    "Report",
    "StudyConfig",
    "Task",
    # Level 2 — typed stores
    "AnalysisBase",
    "AnalysisNode",
    "AnalysisSlice",
    "MAX_TASKS",
    "RegistryEntry",
    "TaskRegistry",
    # Level 2 — firing primitives
    "parallel",
    "retry",
    "rounds",
]
