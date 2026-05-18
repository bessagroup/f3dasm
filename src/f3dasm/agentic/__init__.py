"""Public API for the agentic-f3dasm Level-1 (v2) layer.

The v2 architecture is a thin Python runtime over two persistent
Claude Agent SDK sessions:

- **Strategizer** — reads any file, writes ``.md`` notes only,
  delegates execution to the Implementer.
- **Implementer** — reads/writes/executes freely inside the study tree,
  returns structured reports to the Strategizer.

The user's only required input is ``<study-dir>/briefing.md``.

Public symbols:

- ``MVP_DEFAULT_MODEL`` — Claude model id used by default.
- ``AgenticRun`` — runtime entry point.
- ``Task`` — dataclass sent to the Implementer on each delegation.
- ``Report`` — dataclass produced from the Implementer's response.
- ``AgenticRunError`` — raised for non-recoverable orchestrator errors.
- ``CHECKPOINT_EVERY`` — Implementer-call cadence.
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
    Task,
)

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
    "CHECKPOINT_EVERY",
    "LookupDataGenerator",
    "MVP_DEFAULT_MODEL",
    "Report",
    "Task",
]
