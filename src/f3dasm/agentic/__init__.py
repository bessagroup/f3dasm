"""Public API for the agentic-f3dasm Level-1 (v2) layer.

The v2 architecture is a thin Python runtime over two persistent
Claude Agent SDK sessions:

- **Strategizer** — reads any file, writes ``.md`` notes only,
  delegates execution to the Implementer.
- **Implementer** — reads/writes/executes freely inside the study tree,
  returns structured reports to the Strategizer.

The user's only required input is ``<study-dir>/briefing.md``.

Public symbols are populated as the modules land:

- ``MVP_DEFAULT_MODEL`` — Claude model id used by default.
- ``AgenticRun`` — runtime entry point (G2).
- prompt constants — see ``agent_prompts`` (G1).

Until those land this module exports only the model constant and the
``LookupDataGenerator`` utility that the Implementer may import inside
its own scripts.
"""
#                                                                       Modules
# =============================================================================
from __future__ import annotations

# Local
from .._src.datageneration.lookup import LookupDataGenerator

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Elvis Aguero (elvis_alexander_aguero_vera@brown.edu)"
__credits__ = ["Elvis Aguero"]
__status__ = "Experimental"
# =============================================================================


MVP_DEFAULT_MODEL: str = "claude-haiku-4-5-20251001"


__all__ = [
    "LookupDataGenerator",
    "MVP_DEFAULT_MODEL",
]
