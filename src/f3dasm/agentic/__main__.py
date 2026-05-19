"""CLI entry point for the agentic-f3dasm v2 runtime.

Usage
-----
Run a study from its directory::

    python -m f3dasm.agentic <study-dir>

The study directory must contain a ``PROBLEM_STATEMENT.md`` file.
All runtime parameters come from ``<study-dir>/config.yaml``; CLI flags
override them for one-off runs.

Options
-------
--model <id>          LLM model identifier (overrides config.yaml).
--backend <name>      Backend to use: ``claude`` or ``ollama`` (overrides config.yaml).
--budget HH:MM:SS     Wall-clock time budget (overrides config.yaml).
--checkpoint-every N  Delegations between checkpoints (overrides config.yaml).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

__author__ = "Elvis Aguero (elvis_alexander_aguero_vera@brown.edu)"
__credits__ = ["Elvis Aguero"]
__status__ = "Experimental"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m f3dasm.agentic",
        description=(
            "Run the agentic-f3dasm v2 runtime against a study directory."
        ),
    )
    parser.add_argument(
        "study_dir",
        metavar="study-dir",
        type=Path,
        help="Path to the study directory. Must contain PROBLEM_STATEMENT.md.",
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="MODEL",
        help="LLM model identifier (overrides config.yaml).",
    )
    parser.add_argument(
        "--backend",
        default=None,
        choices=["claude", "ollama"],
        help="Backend to use (overrides config.yaml; default: claude).",
    )
    parser.add_argument(
        "--budget",
        default=None,
        metavar="HH:MM:SS",
        help="Wall-clock time budget (overrides config.yaml).",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        metavar="N",
        dest="checkpoint_every",
        help="Delegations between checkpoints (overrides config.yaml).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    from .._src.agentic.agent_runtime import (
        AgenticRun,
        AgenticRunError,
        StudyConfig,
        _load_study_config,
        _parse_budget,
    )
    from .._src.agentic.backends.claude import CLAUDE_BACKEND

    parser = _build_parser()
    args = parser.parse_args(argv)
    study_dir = Path(args.study_dir)

    # Load file config, then apply CLI overrides.
    cfg = _load_study_config(study_dir)
    if args.model is not None:
        cfg = StudyConfig(
            model=args.model,
            backend=cfg.backend,
            budget=cfg.budget,
            checkpoint_every=cfg.checkpoint_every,
        )
    if args.backend is not None:
        cfg = StudyConfig(
            model=cfg.model,
            backend=args.backend,
            budget=cfg.budget,
            checkpoint_every=cfg.checkpoint_every,
        )
    if args.budget is not None:
        try:
            parsed_budget = _parse_budget(args.budget)
        except ValueError as exc:
            print(f"Error: invalid --budget: {exc}", file=sys.stderr)
            return 1
        cfg = StudyConfig(
            model=cfg.model,
            backend=cfg.backend,
            budget=parsed_budget,
            checkpoint_every=cfg.checkpoint_every,
        )
    if args.checkpoint_every is not None:
        cfg = StudyConfig(
            model=cfg.model,
            backend=cfg.backend,
            budget=cfg.budget,
            checkpoint_every=args.checkpoint_every,
        )

    # Resolve backend.
    if cfg.backend == "ollama":
        from .._src.agentic.backends.ollama import OLLAMA_BACKEND
        backend = OLLAMA_BACKEND
    else:
        backend = CLAUDE_BACKEND

    run = AgenticRun(
        study_dir=study_dir,
        study_config=cfg,
        backend=backend,
    )

    try:
        deliverable_path = run.execute()
    except AgenticRunError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(str(deliverable_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
