"""SLURM resource specifications and cluster configuration."""


#                                                                       Modules
# =============================================================================

# Standard
from __future__ import annotations

from dataclasses import dataclass, field

# Third-party
from omegaconf import DictConfig, OmegaConf

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
        Memory per node (e.g. ``"4G"``). Ignored when
        ``mem_per_cpu`` is set (see ``mem_per_cpu``).
    cpus_per_task : int
        Number of CPUs per task.
    nodes : int
        Number of nodes. When ``mem_per_cpu`` is set and ``nodes``
        is left at its default of ``1``, the ``--nodes`` directive
        is omitted from the rendered script (per-task allocation
        makes it redundant); an explicit ``nodes > 1`` is always
        emitted.
    max_array_size : int
        Maximum SLURM array size (capped by cluster policy).
    max_concurrent : int
        Maximum number of concurrently running array tasks.
    max_jobs_per_task : int | None
        Maximum number of open jobs a single array task evaluates
        sequentially. With the default ``1``, the declared ``time``
        and ``mem`` describe the cost of exactly one experiment,
        independent of how many open jobs the step has: when the
        open-job count exceeds ``max_array_size *
        max_jobs_per_task``, the step is submitted as multiple
        sequential waves of array jobs. Set to ``None`` to restore
        a single array submission in which each task evaluates the
        unbounded strided slice of the open jobs (the declared
        resources must then cover ``ceil(n_open /
        max_array_size)`` experiments).
    extra_sbatch : dict[str, str]
        Arbitrary extra ``#SBATCH`` directives as key-value
        pairs.
    ntasks : int
        Number of tasks (``--ntasks``). Always emitted, on every
        cluster. Sites whose ``job_submit`` filter mandates a task
        count (e.g. TU Delft's DelftBlue) reject jobs that omit it;
        ``--ntasks=1`` is harmless where it is not required.
    mem_per_cpu : str | None
        Per-CPU memory (e.g. ``"3968M"``). When set, the rendered
        script emits ``--mem-per-cpu`` instead of ``--mem`` (the two
        are mutually exclusive in SLURM) and ``mem`` is ignored.
        Required by sites enforcing per-CPU memory accounting.
        Defaults to ``None`` (per-node ``--mem`` is used). The
        precedence over ``mem`` is documented, not validated: a
        dataclass cannot tell an explicitly-set ``mem`` from its
        default, so no runtime "both set" check is attempted. Keep
        ``cpus_per_task * mem_per_cpu`` under the partition's
        ``MaxMemPerCPU`` cap — that remains the caller's concern.
    """

    time: str = "01:00:00"
    mem: str = "4G"
    cpus_per_task: int = 1
    nodes: int = 1
    max_array_size: int = 900
    max_concurrent: int = 64
    max_jobs_per_task: int | None = 1
    extra_sbatch: dict[str, str] = field(default_factory=dict)
    ntasks: int = 1
    mem_per_cpu: str | None = None

    def __post_init__(self) -> None:
        if self.max_jobs_per_task is not None and self.max_jobs_per_task < 1:
            raise ValueError(
                "max_jobs_per_task must be a positive integer or None "
                f"(got {self.max_jobs_per_task!r})."
            )


@dataclass
class SlurmCluster:
    """Configuration for a specific SLURM cluster.

    The cluster is assumed to be a POSIX system (Linux): generated
    sbatch scripts use POSIX-style paths and bash shell syntax,
    and ``sbatch`` must be available on the submission host's
    ``PATH``.

    Parameters
    ----------
    partition : str
        SLURM partition name.
    account : str
        SLURM account string.
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
    mem_per_cpu : str | None
        The cluster's per-CPU memory cap (e.g. ``"3968M"``), or
        ``None`` (the default) for clusters that accept the per-node
        ``--mem``. When set, every generated script (the orchestrator
        and each step) emits ``--mem-per-cpu`` instead of ``--mem``,
        deriving the per-CPU value from each resource's declared
        per-node ``mem`` as ``ceil(mem / cpus_per_task)`` and bumping
        ``cpus_per_task`` up to ``ceil(mem / cap)`` when the node's
        core:memory ratio cannot supply the declared memory on the
        requested cores (a resource that sets ``mem_per_cpu``
        explicitly bypasses this and is rendered verbatim). Set for
        sites whose ``job_submit`` filter rejects ``--mem`` and caps
        per-CPU memory (e.g. TU Delft's DelftBlue, ``"3968M"``); leave
        ``None`` on clusters that accept per-node ``--mem`` (e.g.
        Brown's Oscar).
    """

    partition: str = "batch"
    account: str = "default"
    env_setup: list[str] = field(default_factory=list)
    env_vars: dict[str, str] = field(default_factory=dict)
    runner: str = "python"
    log_dir: str = "logs/{project_job}"
    mem_per_cpu: str | None = None

    @classmethod
    def from_yaml(cls, config: DictConfig) -> SlurmCluster:
        """Create a SlurmCluster from a Hydra DictConfig.

        Parameters
        ----------
        config : DictConfig
            Hydra DictConfig for the cluster section, e.g. ``cfg.cluster``.

        Returns
        -------
        SlurmCluster

        Examples
        --------
        >>> cluster = SlurmCluster.from_yaml(cfg.cluster)
        """
        _dict = OmegaConf.to_container(config, resolve=True)
        _dict.pop("enabled", None)
        return cls(**_dict)
