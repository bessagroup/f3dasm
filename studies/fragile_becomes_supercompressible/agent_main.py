"""Entry point for the agentic supercompressible optimisation run.

Usage examples
--------------
Stub mode (no Claude API key required, working path for M7):

    python agent_main.py --stub --iterations 5

Real Claude SDK mode (M8 — send() is not yet implemented):

    python agent_main.py --iterations 10 --model claude-haiku-4-5-20251001

The script exposes a ``main()`` function for programmatic use (e.g. in
tests) alongside the CLI argument parser at the module bottom.

NOTE — ``input_name`` inelegance.
The base ``Optimizer.arm`` interface accepts a single ``input_name``
string. For this multi-input problem we pass ``"ratio_d"`` as a
placeholder; the agentic layer does not use ``input_name`` to generate
candidates — candidates come from the ``StrategyRegistry`` samplers —
so the choice is arbitrary.  A future Level-2 API should accept a list.

NOTE — ClaudeSDKAgent.send() raises NotImplementedError.
The live SDK integration is deferred to M8.  Run with ``--stub`` for a
fully working end-to-end demonstration.
"""
#                                                                       Modules
# =============================================================================
from __future__ import annotations

# Standard
import argparse
import datetime
import sys
from pathlib import Path
from typing import Any

# Resolve the repo root so this script can be run from any cwd.
_STUDY_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _STUDY_DIR.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

# Local — f3dasm core
# Local — study-specific post-processing block
from supercompressible_objective import (  # noqa: E402
    SupercompressibleObjective,
)

from f3dasm import ExperimentData  # noqa: E402

# Local — agentic layer
from f3dasm.agentic import (  # noqa: E402
    ClaudeSDKAgent,
    LookupDataGenerator,
    OutputSpec,
    StrategizerImplementerOptimizer,
    Strategy,
)
from f3dasm.design import Domain  # noqa: E402

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Elvis Aguero (elvis_alexander_aguero_vera@brown.edu)"
__credits__ = ["Elvis Aguero"]
__status__ = "Experimental"
# =============================================================================

# ---------------------------------------------------------------------------
# Dataset and study constants
# ---------------------------------------------------------------------------

# Path to the pre-computed 1000-sample 3D pool.
_POOL_DIR = (
    _STUDY_DIR
    / "experiment_data"
    / "supercompressible_3d"
)

# Input columns in the 3D pool.
_INPUT_COLS = ["ratio_d", "ratio_pitch", "ratio_top_diameter"]

# All output columns produced by the FEM pipeline.
_OUTPUT_COLS = ["coilable", "sigma_crit", "energy", "objective"]

# Bounds for the three free design variables (from 3d_domain.yaml).
_BOUNDS = {
    "ratio_d":            (0.004, 0.073),
    "ratio_pitch":        (0.25,  1.50),
    "ratio_top_diameter": (0.0,   0.8),
}

# Constants in the 3D problem (never proposed by the optimizer).
_CONSTANTS = [
    "young_modulus",
    "n_longerons",
    "bottom_diameter",
    "ratio_shear_modulus",
    "circular",
]

# Physics context for the agents — unsolved-problem framing per
# architecture.md ("Recommended physics_context content").  No numerical
# anchor or reference-design mention is included.
_PHYSICS_CONTEXT = (
    "A conical deployable mast made of a brittle polymer (PLA, fracture "
    "strain ~2%) is compressed along its axis.  For most geometries the "
    "mast buckles and breaks; for a specific region of the three-parameter "
    "design space it instead coils reversibly along its axis, achieving "
    "near-complete collapse and springing back to its original shape.  "
    "Three continuous geometric ratios control the design: ratio_d "
    "(longeron cross-section / bottom diameter), ratio_pitch "
    "(height / bottom diameter), and ratio_top_diameter "
    "(taper: (D1-D2)/D1).  The scalar objective to maximise is sigma_crit "
    "(critical buckling stress, kPa) restricted to designs where the "
    "coilable output equals 1 (reversibly coilable); designs where coilable "
    "is 0 (wrong buckling mode) or 2 (coils but fractures) receive a large "
    "negative penalty.  The coilable=2 class is geometrically adjacent to "
    "coilable=1 — reducing ratio_d often moves a coilable=2 design into the "
    "feasible region."
)

# Typed metadata for every output column shown to the agents.
_OUTPUT_COLUMNS: dict[str, OutputSpec] = {
    "coilable": OutputSpec(
        kind="categorical",
        categories=[0, 1, 2],
        description=(
            "0 = wrong buckling mode; "
            "1 = reversibly coilable; "
            "2 = coils but fractures"
        ),
    ),
    "sigma_crit": OutputSpec(
        kind="continuous",
        description="critical buckling stress, kPa",
    ),
    "energy": OutputSpec(
        kind="continuous",
        description="elastic energy absorption, kJ/m^3 (stochastic)",
    ),
    "objective": OutputSpec(
        kind="continuous",
        description="scalar objective: sigma_crit if coilable=1 else -1e6",
    ),
}


# =============================================================================
# StubAgent — used when --stub flag is set
# =============================================================================


class _StubAgent:
    """Local test double that exercises the tool API without a live LLM.

    The Strategizer stub emits a ``latin`` strategy; the Implementer stub
    calls ``run_strategy`` with it.  Behaviour mirrors the pattern in
    ``tests/optimization/test_agent_optimizer.py``.

    Parameters
    ----------
    name : str
        Agent name (``'strategizer'`` or ``'implementer'``).
    system_prompt : str
        Stored but not used by the stub.
    """

    def __init__(self, name: str, system_prompt: str, **kwargs: Any) -> None:
        """Initialise the stub.

        Parameters
        ----------
        name : str
            Agent role identifier.
        system_prompt : str
            System prompt (unused by the stub).
        **kwargs : Any
            Accepted for factory-call compatibility; ignored.
        """
        self.name = name
        self._system_prompt = system_prompt

    def send(self, message: str, tools: list) -> str:
        """Interact with the exposed tools and return a rationale response.

        Parameters
        ----------
        message : str
            Orchestrator turn message; ignored by the stub.
        tools : list
            Tool callables exposed for this turn.

        Returns
        -------
        str
            Response text containing a ``## Rationale`` section.
        """
        # Build a name-to-callable map.
        tool_map: dict[str, Any] = {t.__name__: t for t in tools}

        if self.name == "strategizer":
            # Emit a Latin-hypercube strategy to explore the space.
            if "emit_strategy" in tool_map:
                strategy = Strategy(
                    name="latin",
                    n_steps=5,
                    params={},
                    intent=(
                        "Use Latin hypercube sampling to explore the "
                        "three-dimensional design space broadly.  The goal "
                        "is to discover coilable=1 regions by spreading "
                        "evaluations across the full parameter range."
                    ),
                )
                tool_map["emit_strategy"](
                    {
                        "name": strategy.name,
                        "n_steps": strategy.n_steps,
                        "params": strategy.params,
                        "intent": strategy.intent,
                    }
                )
            return (
                "## Rationale\n"
                "Stub Strategizer: emitted a latin strategy with 5 steps "
                "to explore the supercompressible design space."
            )
        else:
            # Implementer: run the strategy.
            if "run_strategy" in tool_map:
                tool_map["run_strategy"]("latin", 5, {})
            return (
                "## Rationale\n"
                "Stub Implementer: executed latin strategy with 5 steps."
            )


def _stub_strategizer_factory(
    name: str, system_prompt: str, **kwargs: Any
) -> _StubAgent:
    """Return a Strategizer ``_StubAgent``.

    Parameters
    ----------
    name : str
        Agent name.
    system_prompt : str
        System prompt.
    **kwargs : Any
        Extra arguments accepted for compatibility.

    Returns
    -------
    _StubAgent
        Configured as a Strategizer.
    """
    return _StubAgent(name=name, system_prompt=system_prompt)


def _stub_implementer_factory(
    name: str, system_prompt: str, **kwargs: Any
) -> _StubAgent:
    """Return an Implementer ``_StubAgent``.

    Parameters
    ----------
    name : str
        Agent name.
    system_prompt : str
        System prompt.
    **kwargs : Any
        Extra arguments accepted for compatibility.

    Returns
    -------
    _StubAgent
        Configured as an Implementer.
    """
    return _StubAgent(name=name, system_prompt=system_prompt)


# =============================================================================
# Domain builder
# =============================================================================


def _build_live_domain() -> Domain:
    """Build the optimizer's view of the 3D design domain.

    Returns
    -------
    Domain
        Contains the three continuous free inputs and four output columns.
    """
    domain = Domain()
    for param_name, (low, high) in _BOUNDS.items():
        domain.add_float(param_name, low=low, high=high)
    # Register all output columns used during the run.
    for col in _OUTPUT_COLS:
        domain.add_output(col, to_disk=False, exist_ok=True)
    return domain


# =============================================================================
# Pool ceiling helper
# =============================================================================


def _pool_ceiling(pool: ExperimentData) -> float:
    """Return the maximum sigma_crit among coilable=1 rows in the pool.

    Parameters
    ----------
    pool : ExperimentData
        The full 1000-sample pre-computed dataset.

    Returns
    -------
    float
        Maximum sigma_crit among coilable=1 rows.  Returns ``float('-inf')``
        if no coilable=1 rows exist.
    """
    best = float("-inf")
    for sample in pool.data.values():
        if int(sample._output_data.get("coilable", 0)) != 1:
            continue
        sc = sample._output_data.get("sigma_crit")
        if sc is None:
            continue
        try:
            sc_f = float(sc)
        except (TypeError, ValueError):
            continue
        if sc_f > best:
            best = sc_f
    return best


# =============================================================================
# Best-design extractor
# =============================================================================


def _best_coilable_1(live: ExperimentData) -> dict:
    """Find the best coilable=1 design in the live dataset.

    Parameters
    ----------
    live : ExperimentData
        The optimizer's working dataset after ``call()``.

    Returns
    -------
    dict
        Keys: ``inputs`` (dict), ``sigma_crit`` (float), ``objective``
        (float).  Empty dict if no coilable=1 row exists.
    """
    best_obj = float("-inf")
    best_row: dict = {}
    for sample in live.data.values():
        coilable = sample._output_data.get("coilable")
        if coilable is None or int(coilable) != 1:
            continue
        obj = sample._output_data.get("objective", float("-inf"))
        try:
            obj_f = float(obj)
        except (TypeError, ValueError):
            continue
        if obj_f > best_obj:
            best_obj = obj_f
            best_row = {
                "inputs": dict(sample._input_data),
                "sigma_crit": sample._output_data.get("sigma_crit"),
                "objective": obj_f,
            }
    return best_row


# =============================================================================
# main()
# =============================================================================


def main(
    stub: bool = True,
    iterations: int = 10,
    model: str = "claude-haiku-4-5-20251001",
    project_dir: Path | None = None,
) -> dict:
    """Run the agentic supercompressible optimisation loop.

    Parameters
    ----------
    stub : bool, optional
        If ``True``, use ``_StubAgent`` instead of ``ClaudeSDKAgent``.
        Default is ``True``.
    iterations : int, optional
        Number of agentic iterations to run.  Default is 10.
    model : str, optional
        Claude model identifier (used only when ``stub=False``).
    project_dir : Path or None, optional
        Directory for ``turn_log.jsonl`` and any persisted artefacts.
        Created if it does not exist.  Defaults to a timestamped
        subdirectory under ``studies/.../runs/``.

    Returns
    -------
    dict
        Summary with keys:
        - ``"best_coilable_1"`` — dict with ``inputs``, ``sigma_crit``,
          ``objective`` of the best coilable=1 design found, or ``{}`` if
          none found.
        - ``"ceiling"`` — maximum sigma_crit among coilable=1 pool rows.
        - ``"turn_log_path"`` — ``Path`` to the written ``turn_log.jsonl``.
        - ``"n_evaluated"`` — number of rows in the live dataset after the
          run.
        - ``"deliverable_path"`` — ``Path`` to the written deliverable
          folder.
    """
    # ------------------------------------------------------------------
    # Resolve project directory
    # ------------------------------------------------------------------
    if project_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        project_dir = _STUDY_DIR / "runs" / timestamp
    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load the 1000-sample pool
    # ------------------------------------------------------------------
    pool = ExperimentData.from_file(_POOL_DIR)

    # ------------------------------------------------------------------
    # Apply SupercompressibleObjective to the pool rows
    # ------------------------------------------------------------------
    objective_block = SupercompressibleObjective()
    pool = objective_block.call(pool)

    # ------------------------------------------------------------------
    # Compute the pool ceiling for comparison
    # ------------------------------------------------------------------
    ceiling = _pool_ceiling(pool)

    # ------------------------------------------------------------------
    # Build the live dataset (empty rows, same 3D domain)
    # ------------------------------------------------------------------
    live_domain = _build_live_domain()
    live = ExperimentData(domain=live_domain, project_dir=project_dir)

    # ------------------------------------------------------------------
    # Build the LookupDataGenerator over the pool
    # ------------------------------------------------------------------
    lookup_gen = LookupDataGenerator(
        pool=pool,
        input_columns=_INPUT_COLS,
        output_columns=_OUTPUT_COLS,
    )

    # ------------------------------------------------------------------
    # Build the agent factories
    # ------------------------------------------------------------------
    if stub:
        strategizer_factory = _stub_strategizer_factory
        implementer_factory = _stub_implementer_factory
    else:
        # NOTE: ClaudeSDKAgent.send() raises NotImplementedError in M7.
        # The real Claude SDK invocation will land in M8.
        def strategizer_factory(name: str, system_prompt: str,
                                **kw: Any) -> ClaudeSDKAgent:
            return ClaudeSDKAgent(
                name=name, system_prompt=system_prompt, model=model
            )

        def implementer_factory(name: str, system_prompt: str,
                                **kw: Any) -> ClaudeSDKAgent:
            return ClaudeSDKAgent(
                name=name, system_prompt=system_prompt, model=model
            )

    # ------------------------------------------------------------------
    # Construct the StrategizerImplementerOptimizer
    # ------------------------------------------------------------------
    optimizer = StrategizerImplementerOptimizer(
        strategizer_factory=strategizer_factory,
        implementer_factory=implementer_factory,
        physics_context=_PHYSICS_CONTEXT,
        output_columns=_OUTPUT_COLUMNS,
        max_followups=2,
    )

    # ------------------------------------------------------------------
    # Arm the optimizer
    # NOTE: input_name is a single-string API limitation; we pass
    # "ratio_d" as a placeholder.  The strategy samplers iterate over
    # all variable_parameters from the Domain, so the actual input used
    # for candidate generation is not restricted to this one column.
    # ------------------------------------------------------------------
    optimizer.arm(
        data=live,
        data_generator=lookup_gen,
        input_name="ratio_d",
        output_name="objective",
    )

    # ------------------------------------------------------------------
    # Run the agentic loop
    # ------------------------------------------------------------------
    live = optimizer.call(live, n_iterations=iterations)

    # ------------------------------------------------------------------
    # Compute summary
    # ------------------------------------------------------------------
    best = _best_coilable_1(live)
    turn_log_path = project_dir / "turn_log.jsonl"
    n_evaluated = len(live.data)

    # ------------------------------------------------------------------
    # Write deliverable folder
    # ------------------------------------------------------------------
    deliverable_dir = project_dir / "deliverable"
    deliverable_path = optimizer.write_deliverable(
        out_dir=deliverable_dir,
        ceiling_value=ceiling if ceiling != float("-inf") else None,
    )

    # ------------------------------------------------------------------
    # Print summary to stdout
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Agentic Supercompressible Run — Summary")
    print("=" * 60)
    print(f"  Iterations requested : {iterations}")
    print(f"  Rows evaluated       : {n_evaluated}")
    print(f"  Turn log             : {turn_log_path}")
    print(f"  Deliverable folder   : {deliverable_path}")
    print(f"  Pool ceiling σ_crit  : {ceiling:.6g} kPa (coilable=1)")
    if best:
        print("  Best coilable=1 found:")
        for k, v in best["inputs"].items():
            print(f"    {k:25s}: {v:.6g}")
        print(f"    {'sigma_crit':25s}: {best['sigma_crit']:.6g} kPa")
        print(f"    {'objective':25s}: {best['objective']:.6g}")
    else:
        print("  Best coilable=1 found: (none)")
    print("=" * 60 + "\n")

    return {
        "best_coilable_1": best,
        "ceiling": ceiling,
        "turn_log_path": turn_log_path,
        "n_evaluated": n_evaluated,
        "deliverable_path": deliverable_path,
    }


# =============================================================================
# CLI entry point
# =============================================================================


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : list of str or None
        Argument list; ``None`` reads from ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Agentic supercompressible optimisation entry point."
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of agentic iterations to run (default: 10).",
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help=(
            "Use StubAgent instead of ClaudeSDKAgent.  "
            "Required for M7 testing (ClaudeSDKAgent.send() is not yet "
            "implemented)."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-haiku-4-5-20251001",
        help="Claude model identifier (used only when --stub is not set).",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=None,
        dest="project_dir",
        help=(
            "Directory for turn_log.jsonl and persisted artefacts. "
            "Defaults to studies/.../runs/<timestamp>/."
        ),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    main(
        stub=args.stub,
        iterations=args.iterations,
        model=args.model,
        project_dir=args.project_dir,
    )
