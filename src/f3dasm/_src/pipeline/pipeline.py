"""Pipeline and Step definitions for composing f3dasm blocks."""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

# Local
from .loop import Loop
from .resources import SlurmCluster, SlurmResources

if TYPE_CHECKING:
    from ..core import Block, DataGenerator

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================
#
# =============================================================================


logger = logging.getLogger("f3dasm")

# Valid SLURM dependency types for Step.dependency
VALID_DEPENDENCIES = ("afterok", "afterany")

# =============================================================================


@dataclass
class Step:
    """A single step in a :class:`Pipeline`.

    Parameters
    ----------
    block : Block | DataGenerator | Callable
        The operation to execute. Can be a :class:`Block`,
        :class:`DataGenerator`, or a plain callable with signature
        ``(project_dir, project_job, **kwargs) -> None``.
    name : str
        Human-readable name for this step (used in logs and SLURM
        job names).
    parallel : bool
        If ``True``, this step is executed as a SLURM array job
        (or with multiprocessing locally). Only meaningful when
        the block is a :class:`DataGenerator`.
    resources : SlurmResources
        SLURM resource requirements for this step.
    dependency : Literal["afterok", "afterany"]
        SLURM dependency type for the previous step.
        Must be ``"afterok"`` or ``"afterany"``.
    array_jobs : int, optional
        If ``parallel=True``, the number of array jobs to submit.
        Required if the block is a :class:`DataGenerator`.
    project_dir : str
        Sub-path relative to the job directory where
        ExperimentData is loaded and stored for this step.
        Defaults to ``'.'`` (the job directory itself).
    kwargs : dict[str, Any]
        Extra keyword arguments forwarded to the block's
        ``call`` method at execution time.
    """

    block: Block | DataGenerator | Callable
    name: str = ""
    parallel: bool = False
    resources: SlurmResources = field(default_factory=SlurmResources)
    dependency: Literal["afterok", "afterany"] = "afterok"
    array_jobs: int | None = None
    project_dir: str = "."
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.dependency not in VALID_DEPENDENCIES:
            raise ValueError(
                f"Invalid dependency {self.dependency!r}. "
                f"Must be one of {VALID_DEPENDENCIES}."
            )


# Type alias for pipeline step entries
PipelineElement = Step | Loop


@dataclass
class Pipeline:
    """A composable, executable pipeline of f3dasm blocks.

    A pipeline is an ordered sequence of :class:`Step` and
    :class:`Loop` objects. Steps run once; loops repeat their
    inner steps for a given number of iterations.

    In SLURM mode, a single self-resubmitting orchestrator script
    manages the entire pipeline, submitting one step (or one loop
    iteration) at a time.

    Parameters
    ----------
    name : str
        Name of the pipeline (used for directory naming and SLURM
        job prefixes).
    steps : list[Step | Loop]
        The ordered sequence of steps and loops.
    orchestrator_resources : SlurmResources, optional
        SLURM resource requirements for the orchestrator job. If
        ``None``, a lightweight default is used (10 min, 1 GB,
        1 CPU). The orchestrator only runs ``sbatch`` commands, so
        minimal resources suffice.

    Examples
    --------
    Simple three-phase pipeline::

        pipeline = Pipeline(
            name="my_experiment",
            steps=[
                Step("create", block=my_create_block),
                Step("run", block=my_generator, parallel=True),
                Step("post", block=my_post_block),
            ],
        )
        pipeline.run(mode="local")

    Pipeline with a loop::

        pipeline = Pipeline(
            name="online_rl",
            steps=[
                Step("create", block=create_block),
                Loop(n_iterations=10, steps=[
                    Step("run", block=generator, parallel=True),
                    Step("post", block=update_block),
                ]),
            ],
        )
        pipeline.run(
            mode="slurm",
            cluster=SlurmCluster(
                partition="batch",
                account="my_account",
            ),
            rootdir="/scratch/user",
        )
    """

    name: str = ""
    steps: list[PipelineElement] = field(default_factory=list)
    orchestrator_resources: SlurmResources | None = None

    def _flatten(self) -> list[tuple[Step, int, int]]:
        """Flatten the pipeline into an ordered list of steps.

        Walks the ``steps`` list and expands any :class:`Loop`
        elements into repeated step entries.

        Returns
        -------
        list[tuple[Step, int, int]]
            Each entry is ``(step, iteration, n_iterations)``
            where ``iteration`` is the current loop iteration
            (0 for non-looped steps) and ``n_iterations`` is the
            total (1 for non-looped steps).
        """
        flat: list[tuple[Step, int, int]] = []
        for element in self.steps:
            if isinstance(element, Step):
                flat.append((element, 0, 1))
            elif isinstance(element, Loop):
                for i in range(element.n_iterations):
                    for step in element.steps:
                        flat.append((step, i, element.n_iterations))
        return flat

    def run(
        self,
        mode: Literal["local", "slurm"] = "local",
        cluster: SlurmCluster | None = None,
        project_job: str | None = None,
        rootdir: Path | str | None = None,
    ) -> str:
        """Execute the pipeline.

        Parameters
        ----------
        mode : Literal["local", "slurm"]
            Execution mode: ``"local"`` or ``"slurm"``.
        cluster : SlurmCluster, optional
            Cluster configuration (required when
            ``mode="slurm"``).
        project_job : str, optional
            Job identifier used as the top-level run folder
            (``rootdir / project_job``). Defaults to
            ``str(int(time.time()))``. Pass an existing ID to
            resume a previous run.
        rootdir : Path | str, optional
            Root directory under which the job folder is created.
            Defaults to the current working directory.

        Returns
        -------
        str
            The project job ID.
        """
        # Lazy imports to avoid circular dependency:
        # pipeline -> executors -> pipeline
        from .executors.local import LocalExecutor
        from .executors.slurm import SlurmExecutor

        _rootdir: Path | None = Path(rootdir) if rootdir is not None else None

        if mode == "local":
            executor = LocalExecutor()
        elif mode == "slurm":
            if cluster is None:
                raise ValueError(
                    "A SlurmCluster must be provided for mode='slurm'."
                )
            executor = SlurmExecutor(cluster=cluster)
        else:
            raise ValueError(f"Unknown mode {mode!r}. Use 'local' or 'slurm'.")

        return executor.run(
            pipeline=self,
            project_job=project_job,
            rootdir=_rootdir,
        )

    def generate_scripts(
        self,
        cluster: SlurmCluster,
        project_job: str = "PLACEHOLDER",
        rootdir: Path | str | None = None,
    ) -> dict[str, str]:
        """Generate SLURM scripts without submitting them.

        Useful for inspecting or manually editing scripts before
        submission. For pipelines containing loops, the result
        includes orchestrator scripts and loop body step scripts
        with ``$F3DASM_ITERATION`` as the iteration placeholder.

        Parameters
        ----------
        cluster : SlurmCluster
            Cluster configuration.
        project_job : str
            Project job ID to embed in scripts.
        rootdir : Path | str, optional
            Root directory under which the job folder is created.
            Defaults to the current working directory.

        Returns
        -------
        dict[str, str]
            Mapping of label to rendered sbatch script content.
        """
        from .executors.slurm import SlurmExecutor

        _rootdir: Path | None = Path(rootdir) if rootdir is not None else None
        executor = SlurmExecutor(cluster=cluster)
        return executor.generate_scripts(
            pipeline=self,
            project_job=project_job,
            rootdir=_rootdir,
        )
