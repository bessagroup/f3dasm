"""CLI helper that prints the number of open ExperimentData rows.

Used by the SLURM orchestrator to size the ``--array=`` directive
at submission time, after the upstream step has finished writing
its ExperimentData to disk. Prints a single integer to stdout.

Usage::

    python -m f3dasm._src.pipeline.count_open \\
        --job-dir=/scratch/user/1711449600 \\
        --project-dir=.
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
import argparse
import sys
from pathlib import Path

# Local
from ..experimentdata import ExperimentData

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================
#
# =============================================================================


def main(argv: list[str] | None = None) -> None:
    """Print the count of open experiments in an ExperimentData."""
    parser = argparse.ArgumentParser(
        description="Print the number of open experiments on disk."
    )
    parser.add_argument(
        "--job-dir",
        required=True,
        type=str,
        help="Absolute path to the job directory (rootdir/project_job).",
    )
    parser.add_argument(
        "--project-dir",
        required=False,
        type=str,
        default=".",
        help="Relative sub-path within job_dir where ExperimentData lives.",
    )

    args = parser.parse_args(argv)
    run_dir: Path = Path(args.job_dir) / args.project_dir
    data: ExperimentData = ExperimentData.from_file(project_dir=run_dir)
    n_open: int = len(data.select_with_status("open"))
    sys.stdout.write(f"{n_open}\n")


if __name__ == "__main__":
    main()
