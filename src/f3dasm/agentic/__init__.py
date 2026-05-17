"""Public API for the agentic-f3dasm Level-1 layer.

Import from this module rather than the underlying ``_src/...`` paths.
The agentic subpackage is strictly additive: nothing here modifies the
core f3dasm objects.  Downstream studies and notebooks should use::

    from f3dasm.agentic import ProblemSchema, Strategy, default_registry, ...

The symbols re-exported here are the complete Level-1 public surface.
"""
#                                                                       Modules
# =============================================================================
from __future__ import annotations

# Local — Level-1 evaluator
from f3dasm._src.datageneration.lookup import LookupDataGenerator

# Local — dataclasses backing the agentic layer
from f3dasm._src.optimization.agent_dataclasses import (
    OutputSpec,
    ParameterSpec,
    ParamSignature,
    ProblemSchema,
    Strategy,
    StrategySpec,
    TurnRecord,
)

# Local — optimizer and agent classes (M6)
from f3dasm._src.optimization.agent_optimizer import (
    Agent,
    AgentExecutionError,
    AgentOptimizer,
    ClaudeSDKAgent,
    StrategizerImplementerOptimizer,
)

# Local — artifact producers for AnalysisBase
from f3dasm._src.optimization.artifact_producers import (
    coverage_summary,
    last_5_rationales,
    objective_summary,
)

# Local — strategy registry and utilities
from f3dasm._src.optimization.strategies import (
    StrategyAdapter,
    StrategyOutcome,
    default_registry,
    validate_params,
)

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Elvis Aguero (elvis_alexander_aguero_vera@brown.edu)"
__credits__ = ["Elvis Aguero"]
__status__ = "Experimental"
# =============================================================================

# Default Claude model identifier for Level-1 studies.
MVP_DEFAULT_MODEL: str = "claude-haiku-4-5-20251001"

__all__ = [
    # agent_dataclasses
    "ParameterSpec",
    "OutputSpec",
    "ProblemSchema",
    "ParamSignature",
    "StrategySpec",
    "Strategy",
    "TurnRecord",
    # lookup
    "LookupDataGenerator",
    # strategies
    "StrategyAdapter",
    "StrategyOutcome",
    "default_registry",
    "validate_params",
    # artifact_producers
    "objective_summary",
    "coverage_summary",
    "last_5_rationales",
    # agent_optimizer (M6)
    "Agent",
    "AgentExecutionError",
    "AgentOptimizer",
    "ClaudeSDKAgent",
    "StrategizerImplementerOptimizer",
    # constants
    "MVP_DEFAULT_MODEL",
]
