"""Dataclasses backing the agentic optimization layer.

This module defines the typed objects exchanged across the agentic-f3dasm
architecture: the problem description shown to every agent
(``ProblemSchema``), the entries of the strategy registry (``StrategySpec``),
the inter-agent communication payload (``Strategy``), and the provenance
records appended to ``TurnLog`` after every agent session (``TurnRecord``).

The classes intentionally carry no f3dasm-specific imports — they are plain
data containers that can be constructed, validated, and round-tripped to JSON.
"""
#                                                                       Modules
# =============================================================================
from __future__ import annotations

# Standard
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Elvis Aguero (elvis_alexander_aguero_vera@brown.edu)"
__credits__ = ["Elvis Aguero"]
__status__ = "Experimental"
# =============================================================================


# Type aliases used throughout the module.
ParameterKind = Literal["continuous", "discrete", "categorical"]
OutputKind = Literal["continuous", "categorical"]
ObjectiveDirection = Literal["minimize", "maximize"]
ParamType = Literal[
    "int", "float", "bool", "str", "list[float]", "list[int]"
]
TurnResult = Literal["ok", "error"]


@dataclass(frozen=True)
class ParameterSpec:
    """Typed description of an input parameter exposed to the agent.

    Parameters
    ----------
    kind : {'continuous', 'discrete', 'categorical'}
        The parameter family. ``'discrete'`` covers integer-grid parameters.
    bounds : tuple of (float, float), optional
        Inclusive lower / upper bounds for continuous and discrete parameters.
        ``None`` for categorical parameters.
    categories : list of object, optional
        Allowed values for categorical parameters. ``None`` for continuous
        and discrete parameters.
    log_scale : bool, default False
        Whether continuous parameters should be sampled on a log scale.
        Ignored for non-continuous parameters.
    """

    kind: ParameterKind
    bounds: Optional[tuple[float, float]] = None
    categories: Optional[list[Any]] = None
    log_scale: bool = False


@dataclass(frozen=True)
class OutputSpec:
    """Typed description of an output column the agent reasons about.

    f3dasm's ``Domain`` does not carry type metadata for outputs; this spec
    is supplied explicitly by the study so the agent can interpret outputs
    correctly (e.g. distinguish categorical levels from continuous values).

    Parameters
    ----------
    kind : {'continuous', 'categorical'}
        Whether the output is continuous or categorical.
    categories : list of object, optional
        For categorical outputs, the allowed values. ``None`` otherwise.
    description : str, default ''
        Short human-readable note shown to the agent in the system prompt.
    """

    kind: OutputKind
    categories: Optional[list[Any]] = None
    description: str = ""


@dataclass(frozen=True)
class ProblemSchema:
    """The immutable problem description shown to every agent each turn.

    Built once during ``AgentOptimizer.arm`` from the live ``Domain``, the
    user-supplied ``physics_context`` text, and the per-output ``OutputSpec``
    map. Passed by deepcopy to every agent-facing tool so the live ``Domain``
    cannot be mutated through it.

    Parameters
    ----------
    variable_parameters : dict[str, ParameterSpec]
        Input parameters the agent's strategies may seed. Derived from the
        ``Domain`` minus the forbidden names.
    forbidden_parameters : list of str
        Input parameter names that must never be seeded — typically the
        constant parameters in ``Domain`` plus any user-supplied additions.
    objective_name : str
        Name of the output column the agent maximizes or minimizes.
    objective_direction : {'minimize', 'maximize'}
        Direction of optimization for ``objective_name``.
    output_columns : dict[str, OutputSpec]
        Typed metadata for every output column the agent should know about.
    physics_context : str, default ''
        Day-1 problem briefing in natural language — what a domain expert
        would tell a new lab member before they start working on an
        *unsolved* version of the problem.
    """

    variable_parameters: dict[str, ParameterSpec]
    forbidden_parameters: list[str]
    objective_name: str
    objective_direction: ObjectiveDirection
    output_columns: dict[str, OutputSpec]
    physics_context: str = ""


@dataclass(frozen=True)
class ParamSignature:
    """Typed signature for one parameter of a registered strategy.

    Parameters
    ----------
    type : {'int', 'float', 'bool', 'str', 'list[float]', 'list[int]'}
        Coarse runtime type tag. Used by ``run_strategy`` to validate the
        agent's parameter dict before dispatch.
    required : bool
        Whether this parameter must be present in the agent's call.
    default : object, optional
        Default value when ``required`` is False. ``None`` if there is no
        default.
    description : str
        One-line description shown to the agent via ``list_strategies``.
    """

    type: ParamType
    required: bool
    default: Optional[Any]
    description: str


@dataclass(frozen=True)
class StrategySpec:
    """An entry in the ``StrategyRegistry``.

    Describes one strategy the Implementer may invoke via ``run_strategy``.
    The ``parameters`` mapping lets the orchestrator validate agent calls
    before dispatch.

    Parameters
    ----------
    name : str
        Stable key the Implementer passes to ``run_strategy``.
    description : str
        One-line plain-English summary the Strategizer reads.
    parameters : dict[str, ParamSignature]
        Per-parameter typed signature.
    """

    name: str
    description: str
    parameters: dict[str, ParamSignature]


@dataclass(frozen=True)
class Strategy:
    """The communication payload routed from Strategizer to Implementer.

    Parameters
    ----------
    name : str
        Must match a ``StrategySpec.name`` in the active registry.
    n_steps : int
        Step budget to pass to ``run_strategy``.
    params : dict
        Concrete parameter values, validated against the matching
        ``StrategySpec`` before dispatch.
    intent : str
        Natural-language briefing: *why* this strategy, what region or
        property to focus on, what constraints to respect, what to report
        back. Embeds context, boundaries, contracts, and any operational
        detail the Strategizer wants the Implementer to honor.
    """

    name: str
    n_steps: int
    params: dict[str, Any]
    intent: str


@dataclass
class TurnRecord:
    """A single entry in ``TurnLog`` — the provenance atom.

    One ``TurnRecord`` is appended for every agent session, whether the
    session succeeded (``result='ok'``) or completed without a parseable
    rationale (``result='error'``). Implementer records are the only ones
    that may carry non-empty ``strategy_calls`` and ``experiment_ids``.

    Parameters
    ----------
    turn_id : int
        Monotonically increasing counter assigned at append time.
    timestamp : str
        ISO-8601 wall-clock time when the turn started.
    agent_name : str
        Key into ``AgentOptimizer.agents`` (e.g. ``'strategizer'``).
    rationale : str
        Text extracted from the ``## Rationale`` section of the agent's
        final message. Empty when ``result == 'error'``.
    strategy_calls : list[tuple[str, int, dict]]
        ``(name, n_steps, params)`` triples, one per ``run_strategy`` call
        made during this turn. Only the Implementer ever populates this.
    emitted_strategy : Strategy, optional
        Set when the Strategizer called ``emit_strategy`` this turn.
        ``None`` otherwise.
    followup_question : str, optional
        Set when the Implementer called ``ask_strategizer`` instead of (or
        in addition to) ``run_strategy``. ``None`` otherwise.
    experiment_ids : list of int
        Row indices added to ``ExperimentData`` during this turn (rows
        whose ``__turn`` column equals ``turn_id``).
    result : {'ok', 'error'}
        ``'ok'`` if the session returned a parseable rationale, ``'error'``
        otherwise.
    transcript_ref : str, optional
        Relative path to the session transcript on disk. ``None`` for
        Level 1 (transcript persistence is a Level-2 concern).
    """

    turn_id: int
    timestamp: str
    agent_name: str
    rationale: str
    strategy_calls: list[tuple[str, int, dict[str, Any]]] = field(
        default_factory=list
    )
    emitted_strategy: Optional[Strategy] = None
    followup_question: Optional[str] = None
    experiment_ids: list[int] = field(default_factory=list)
    result: TurnResult = "ok"
    transcript_ref: Optional[str] = None

    def to_json(self) -> str:
        """Serialize to a one-line JSON string for ``turn_log.jsonl``.

        Returns
        -------
        str
            JSON object encoding the record. The embedded ``Strategy`` (if
            any) is flattened into a nested object.
        """

        payload = asdict(self)
        # asdict converts dataclass fields recursively; tuples in
        # strategy_calls survive as lists, which is the desired wire shape.
        return json.dumps(payload, default=_json_default)

    @classmethod
    def from_json(cls, line: str) -> TurnRecord:
        """Deserialize a one-line JSON string back into a ``TurnRecord``.

        Parameters
        ----------
        line : str
            A single line as produced by :meth:`to_json`.

        Returns
        -------
        TurnRecord
            The reconstructed record.
        """

        raw = json.loads(line)
        emitted = raw.get("emitted_strategy")
        emitted_obj = Strategy(**emitted) if emitted is not None else None
        # JSON encodes (name, n_steps, params) triples as length-3 lists;
        # we convert them back to tuples so downstream typing stays honest.
        strategy_calls = [
            (entry[0], entry[1], entry[2])
            for entry in raw.get("strategy_calls", [])
        ]
        return cls(
            turn_id=raw["turn_id"],
            timestamp=raw["timestamp"],
            agent_name=raw["agent_name"],
            rationale=raw["rationale"],
            strategy_calls=strategy_calls,
            emitted_strategy=emitted_obj,
            followup_question=raw.get("followup_question"),
            experiment_ids=list(raw.get("experiment_ids", [])),
            result=raw.get("result", "ok"),
            transcript_ref=raw.get("transcript_ref"),
        )


def _json_default(obj: Any) -> Any:
    """Fallback JSON encoder for tuples used by ``strategy_calls`` entries.

    Parameters
    ----------
    obj : object
        The value ``json.dumps`` could not encode natively.

    Returns
    -------
    object
        A JSON-serialisable representation.

    Raises
    ------
    TypeError
        If the value is of a type the encoder does not know how to handle.
    """

    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(
        f"Object of type {type(obj).__name__} is not JSON serialisable"
    )


__all__ = [
    "ParameterSpec",
    "OutputSpec",
    "ProblemSchema",
    "ParamSignature",
    "StrategySpec",
    "Strategy",
    "TurnRecord",
]
