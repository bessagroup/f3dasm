"""SLURM resource specifications and cluster configuration."""


#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

from dataclasses import dataclass, field

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================
#
# =============================================================================


@dataclass
class SlurmResources:
    """Resource requirements for a single pipeline step on SLURM.

    Parameters
    ----------
    time : str
        Wall-clock time limit (e.g. ``"01:00:00"``).
    mem : str
        Memory per node (e.g. ``"4G"``).
    cpus_per_task : int
        Number of CPUs per task.
    nodes : int
        Number of nodes.
    max_array_size : int
        Maximum SLURM array size (capped by cluster policy).
    max_concurrent : int
        Maximum number of concurrently running array tasks.
    extra_sbatch : dict[str, str]
        Arbitrary extra ``#SBATCH`` directives as key-value
        pairs.
    """

    time: str = "01:00:00"
    mem: str = "4G"
    cpus_per_task: int = 1
    nodes: int = 1
    max_array_size: int = 900
    max_concurrent: int = 64
    extra_sbatch: dict[str, str] = field(default_factory=dict)


@dataclass
class SlurmCluster:
    """Configuration for a specific SLURM cluster.

    Parameters
    ----------
    partition : str
        SLURM partition name.
    account : str
        SLURM account string.
    scratch_dir : str
        Absolute path to the scratch directory on this cluster.
    env_setup : list[str]
        Shell commands to run before the Python command
        (e.g. module loads, ``unset LD_LIBRARY_PATH``).
    env_vars : dict[str, str]
        Environment variables exported before execution.
    runner : str
        Command prefix for running Python scripts
        (e.g. ``"uv run"`` or ``"python"``).
    log_dir : str
        Log directory template. May contain ``{project_job}``.
    """

    partition: str = "batch"
    account: str = "default"
    scratch_dir: str = "."
    env_setup: list[str] = field(default_factory=list)
    env_vars: dict[str, str] = field(default_factory=dict)
    runner: str = "python"
    log_dir: str = "logs/{project_job}"
