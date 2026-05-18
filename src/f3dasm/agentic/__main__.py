"""CLI entry point for the agentic-f3dasm v2 runtime.

Usage
-----
Run a study from its directory::

    python -m f3dasm.agentic <study-dir>

The study directory must contain a ``PROBLEM_STATEMENT.md`` file.  The agentic
loop runs to completion (or until the user types ``stop`` at a
checkpoint) and then prints the absolute path of the deliverable folder.

Options
-------
--model <id>
    Claude model identifier.  Default: ``claude-haiku-4-5-20251001``.
--checkpoint-every <n>
    Number of Implementer delegations between checkpoints.  Default: 30.
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
import argparse
import sys
from pathlib import Path

# Local
from .._src.agentic.agent_runtime import (
    CHECKPOINT_EVERY,
    MVP_DEFAULT_MODEL,
    AgenticRun,
    AgenticRunError,
)

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Elvis Aguero (elvis_alexander_aguero_vera@brown.edu)"
__credits__ = ["Elvis Aguero"]
__status__ = "Experimental"
# =============================================================================


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
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
        help=(
            "Path to the study directory.  Must contain PROBLEM_STATEMENT.md."
        ),
    )
    parser.add_argument(
        "--model",
        default=MVP_DEFAULT_MODEL,
        metavar="MODEL",
        help=(
            f"Claude model identifier (default: {MVP_DEFAULT_MODEL})."
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=CHECKPOINT_EVERY,
        metavar="N",
        dest="checkpoint_every",
        help=(
            "Number of Implementer delegations between checkpoints "
            f"(default: {CHECKPOINT_EVERY})."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and execute the agentic run.

    Parameters
    ----------
    argv : list[str] or None
        Command-line arguments.  If ``None``, uses ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit code (0 = success, 1 = error).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    run = AgenticRun(
        study_dir=args.study_dir,
        model=args.model,
        checkpoint_every=args.checkpoint_every,
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
