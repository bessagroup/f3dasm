"""Public API for the agentic-f3dasm Level-1 (v2) layer.

The v2 architecture is a thin Python runtime over two persistent
Claude Agent SDK sessions:

- **Strategizer** — reads any file, writes ``.md`` notes only,
  delegates execution to the Implementer.
- **Implementer** — reads/writes/executes freely inside the study tree,
  returns structured reports to the Strategizer.

The user's only required input is ``<study-dir>/PROBLEM_STATEMENT.md``.

Public symbols:

- ``AgenticRun`` — runtime entry point.
- ``Task`` — dataclass sent to the Implementer on each delegation.
- ``Report`` — dataclass produced from the Implementer's response.
- ``AgenticRunError`` — raised for non-recoverable orchestrator errors.
- ``CHECKPOINT_EVERY`` — Implementer-call cadence.
- ``Backend`` — dataclass bundling an LLM backend's factories,
  preflight, and default model.
- ``CLAUDE_BACKEND`` — the default Claude Agent SDK backend.
- ``MVP_DEFAULT_MODEL`` — Claude model id used by default
  (alias of ``CLAUDE_BACKEND.default_model``).
- ``LookupDataGenerator`` — utility the Implementer may import.
- prompt constants — see ``agent_prompts`` module.
"""

#                                                                       Modules
# =============================================================================
from __future__ import annotations

from .._src.agentic.agent_runtime import (
    CHECKPOINT_EVERY,
    MVP_DEFAULT_MODEL,
    AgenticRun,
    AgenticRunError,
    Report,
    StudyConfig,
    Task,
)
from .._src.agentic.backends.base import Backend
from .._src.agentic.backends.claude import CLAUDE_BACKEND
from .._src.agentic.backends.ollama import OLLAMA_BACKEND

# Local
from .._src.agentic.lookup import LookupDataGenerator

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Elvis Aguero (elvis_alexander_aguero_vera@brown.edu)"
__credits__ = ["Elvis Aguero"]
__status__ = "Experimental"
# =============================================================================


__all__ = [
    "AgenticRun",
    "AgenticRunError",
    "Backend",
    "CHECKPOINT_EVERY",
    "CLAUDE_BACKEND",
    "OLLAMA_BACKEND",
    "LookupDataGenerator",
    "MVP_DEFAULT_MODEL",
    "Report",
    "StudyConfig",
    "Task",
]
