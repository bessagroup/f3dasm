"""Strategy adapters and the default Level-1 ``StrategyRegistry``.

A *strategy* in agentic-f3dasm is a named, parameterised primitive that
generates one or more candidate designs, evaluates them through the active
``DataGenerator``, and writes the results into the shared ``ExperimentData``
with provenance (``__turn``) stamped at insertion time. The adapter is the
glue that gives every primitive (Latin sampling, local random search, …) a
uniform call shape so the Implementer's ``run_strategy`` tool can dispatch
any of them by name.

The MVP registry includes only those primitives that work over any
``DataGenerator`` subclass without additional adapter work. Gradient-style
scipy optimizers and Optuna optimizers — which assume the function-wrapping
``data_generator.f`` attribute or a single concatenated input vector — are
deferred to Level 2.
"""
#                                                                       Modules
# =============================================================================
from __future__ import annotations

from collections.abc import Callable

# Standard
from dataclasses import dataclass
from typing import Any, Optional

# Third-party
import numpy as np

# Local
from ..core import DataGenerator
from ..design.domain import Domain
from ..experimentdata import ExperimentData
from ..experimentsample import ExperimentSample
from ..samplers import Grid, Latin, RandomUniform, Sobol
from .agent_dataclasses import ParamSignature, StrategySpec

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Elvis Aguero (elvis_alexander_aguero_vera@brown.edu)"
__credits__ = ["Elvis Aguero"]
__status__ = "Experimental"
# =============================================================================


# A strategy adapter is a callable with the uniform signature below. The
# orchestrator binds the keyword-only arguments at turn time so the
# Implementer's ``run_strategy`` tool only has to forward (name, n_steps,
# params).
StrategyAdapter = Callable[..., "StrategyOutcome"]


@dataclass(frozen=True)
class StrategyOutcome:
    """Result of one strategy invocation.

    Parameters
    ----------
    new_experiment_ids : list of int
        Row indices added to ``ExperimentData`` during this invocation.
    summary : str
        Human-readable summary the tool returns to the agent. Includes
        the rows added, the best objective seen in those rows, and any
        warnings the underlying ``DataGenerator`` surfaced (e.g. pool
        exhaustion).
    """

    new_experiment_ids: list[int]
    summary: str


# =============================================================================
# Helpers
# =============================================================================


def _evaluate_candidates(
    candidates: list[dict[str, Any]],
    *,
    data: ExperimentData,
    data_generator: DataGenerator,
    turn_id: int,
) -> list[int]:
    """Evaluate each candidate and append it to ``data`` with ``__turn`` set.

    Parameters
    ----------
    candidates : list of dict
        Input dicts (column name -> value) for the candidates to evaluate.
    data : ExperimentData
        The persistent dataset the candidates are appended to in place.
    data_generator : DataGenerator
        Evaluator. ``execute`` is called once per candidate.
    turn_id : int
        Current turn index, stamped on every new row's output data under
        the ``__turn`` column.

    Returns
    -------
    list of int
        The indices of the newly inserted rows.
    """

    new_ids: list[int] = []
    for input_data in candidates:
        sample = ExperimentSample(_input_data=dict(input_data))
        sample = data_generator.execute(sample)
        # Stamp provenance before insertion so the row arrives with __turn
        # already set; no post-hoc edit of ExperimentData is required.
        sample._output_data["__turn"] = int(turn_id)
        data.add_experiments(sample, in_place=True)
        new_ids.append(data.index[-1])
    return new_ids


def _summarize_outcome(
    objective_name: str,
    data: ExperimentData,
    new_ids: list[int],
    data_generator: DataGenerator,
) -> str:
    """Build the natural-language summary returned to the agent.

    Parameters
    ----------
    objective_name : str
        Name of the column holding the scalar objective value.
    data : ExperimentData
        The dataset (already updated with the new rows).
    new_ids : list of int
        Indices of the newly inserted rows.
    data_generator : DataGenerator
        The evaluator. If it exposes ``consume_repeats``, the count of
        already-seen pool entries surfaces as a warning.

    Returns
    -------
    str
        The summary text.
    """

    if not new_ids:
        return "No new rows were produced by the strategy."

    new_objectives = []
    for idx in new_ids:
        value = data.data[idx]._output_data.get(objective_name)
        if value is not None:
            new_objectives.append(float(value))

    parts = [f"Added {len(new_ids)} new rows."]
    if new_objectives:
        parts.append(
            "Best objective in this batch: "
            f"{max(new_objectives):.6g} (min: {min(new_objectives):.6g})."
        )
    else:
        parts.append(
            f"No '{objective_name}' values populated on the new rows."
        )

    repeats = 0
    if hasattr(data_generator, "consume_repeats"):
        repeats = data_generator.consume_repeats()
    if repeats:
        parts.append(
            f"WARNING: {repeats} of the {len(new_ids)} candidates landed on "
            "pool entries already evaluated this run; the pool may be "
            "exhausted in the explored region."
        )

    return " ".join(parts)


def _seed_data(domain: Domain) -> ExperimentData:
    """Empty ``ExperimentData`` shell carrying the provided domain.

    The f3dasm samplers need an ``ExperimentData`` argument to read the
    domain; the rows it carries are irrelevant. We construct a fresh empty
    one to avoid mutating the caller's persistent data.
    """

    return ExperimentData(domain=domain)


# =============================================================================
# Adapters
# =============================================================================


def _sampler_adapter(
    sampler_cls: type,
) -> StrategyAdapter:
    """Build an adapter for a f3dasm ``Block``-based sampler class.

    The adapter expects ``n_steps`` to be interpreted as the number of
    samples to draw. An optional ``seed`` parameter is forwarded to the
    sampler.
    """

    def adapter(
        n_steps: int,
        params: dict[str, Any],
        *,
        domain: Domain,
        data: ExperimentData,
        data_generator: DataGenerator,
        objective_name: str,
        turn_id: int,
        **_: Any,
    ) -> StrategyOutcome:
        """Dispatch the wrapped sampler and evaluate each candidate."""

        seed = params.get("seed")
        sampler = sampler_cls(seed=seed)
        seeded = sampler.call(_seed_data(domain), n_samples=n_steps)

        candidates = [
            dict(row._input_data) for row in seeded.data.values()
        ]
        new_ids = _evaluate_candidates(
            candidates,
            data=data,
            data_generator=data_generator,
            turn_id=turn_id,
        )
        summary = _summarize_outcome(
            objective_name=objective_name,
            data=data,
            new_ids=new_ids,
            data_generator=data_generator,
        )
        return StrategyOutcome(new_experiment_ids=new_ids, summary=summary)

    return adapter


def _local_random_adapter(
    n_steps: int,
    params: dict[str, Any],
    *,
    domain: Domain,
    data: ExperimentData,
    data_generator: DataGenerator,
    objective_name: str,
    turn_id: int,
    **_: Any,
) -> StrategyOutcome:
    """Uniformly sample ``n_steps`` candidates inside a local bounding box.

    Parameters
    ----------
    n_steps : int
        Number of candidates to draw.
    params : dict
        Must contain ``center`` (dict[str, float]) and ``radius`` (float)
        keys. ``center`` is a partial input-vector; missing input columns
        default to the column's domain midpoint. ``radius`` is the half-
        width of the box in *normalised* coordinates (0..1 fraction of
        each column's bound range), clipped to the domain bounds.
    domain, data, data_generator, objective_name, turn_id
        Bound by the orchestrator at dispatch time.

    Returns
    -------
    StrategyOutcome
        Indices of new rows and the human-readable summary.
    """

    seed = params.get("seed")
    rng = np.random.default_rng(seed)

    center: dict[str, float] = dict(params.get("center", {}))
    radius = float(params["radius"])
    if not 0.0 < radius <= 1.0:
        raise ValueError(
            f"local_random radius must be in (0, 1], got {radius}."
        )

    candidates: list[dict[str, Any]] = []
    continuous_domain = domain.continuous
    bounds = continuous_domain.get_bounds()
    columns = continuous_domain.input_names

    for _ in range(n_steps):
        candidate: dict[str, Any] = {}
        for col, (low, high) in zip(columns, bounds, strict=False):
            span = high - low
            mid = center.get(col, 0.5 * (low + high))
            half = radius * span
            sample_low = max(low, mid - half)
            sample_high = min(high, mid + half)
            if sample_high <= sample_low:
                sample_high = sample_low + 1e-12
            candidate[col] = float(rng.uniform(sample_low, sample_high))
        candidates.append(candidate)

    new_ids = _evaluate_candidates(
        candidates,
        data=data,
        data_generator=data_generator,
        turn_id=turn_id,
    )
    summary = _summarize_outcome(
        objective_name=objective_name,
        data=data,
        new_ids=new_ids,
        data_generator=data_generator,
    )
    return StrategyOutcome(new_experiment_ids=new_ids, summary=summary)


# =============================================================================
# Registry construction
# =============================================================================


def _seed_signature() -> dict[str, ParamSignature]:
    """Common signature: an optional integer seed."""

    return {
        "seed": ParamSignature(
            type="int",
            required=False,
            default=None,
            description="Random seed for the sampler.",
        ),
    }


_SAMPLER_CLASSES = {
    "latin": (Latin, "Latin hypercube sampling over the variable parameters."),
    "sobol": (
        Sobol,
        "Sobol low-discrepancy sequence over the variable parameters.",
    ),
    "random_uniform": (
        RandomUniform,
        "Uniform random sampling over the variable parameters.",
    ),
    "grid": (
        Grid,
        "Cartesian grid sampling over the variable parameters.",
    ),
}


def default_registry() -> (
    tuple[dict[str, StrategyAdapter], dict[str, StrategySpec]]
):
    """Return the Level-1 strategy adapter and spec maps.

    Returns
    -------
    tuple of (dict, dict)
        The adapter map (strategy name -> callable) and the spec map
        (strategy name -> ``StrategySpec``). The two share keys exactly;
        the orchestrator uses the spec map to validate the agent's
        parameter dict before dispatching the adapter.
    """

    adapters: dict[str, StrategyAdapter] = {}
    specs: dict[str, StrategySpec] = {}

    for name, (sampler_cls, description) in _SAMPLER_CLASSES.items():
        adapters[name] = _sampler_adapter(sampler_cls)
        specs[name] = StrategySpec(
            name=name,
            description=description,
            parameters=_seed_signature(),
        )

    adapters["local_random"] = _local_random_adapter
    specs["local_random"] = StrategySpec(
        name="local_random",
        description=(
            "Uniform random sampling inside a bounding box around a center "
            "point — the exploit primitive."
        ),
        parameters={
            "center": ParamSignature(
                type="str",
                required=True,
                default=None,
                description=(
                    "JSON-like dict of column->value for the box center. "
                    "Missing columns default to their domain midpoint."
                ),
            ),
            "radius": ParamSignature(
                type="float",
                required=True,
                default=None,
                description=(
                    "Half-width of the box as a fraction (0..1] of each "
                    "column's bound range."
                ),
            ),
            "seed": ParamSignature(
                type="int",
                required=False,
                default=None,
                description="Random seed.",
            ),
        },
    )

    return adapters, specs


# =============================================================================
# Param-dict validation
# =============================================================================


def validate_params(
    spec: StrategySpec, params: dict[str, Any]
) -> Optional[str]:
    """Return an error message if ``params`` does not match ``spec``.

    Parameters
    ----------
    spec : StrategySpec
        Spec to validate against.
    params : dict
        Agent-supplied parameters.

    Returns
    -------
    str or None
        ``None`` if the parameters match the signature; otherwise a
        human-readable error message the tool can surface to the agent.
    """

    expected = set(spec.parameters)
    received = set(params)
    unknown = received - expected
    if unknown:
        return (
            f"Unknown parameter(s) for strategy '{spec.name}': "
            f"{sorted(unknown)}. Expected one of {sorted(expected)}."
        )
    for name, signature in spec.parameters.items():
        if signature.required and name not in params:
            return (
                f"Missing required parameter '{name}' for strategy "
                f"'{spec.name}'."
            )
    return None


__all__ = [
    "StrategyAdapter",
    "StrategyOutcome",
    "default_registry",
    "validate_params",
]
