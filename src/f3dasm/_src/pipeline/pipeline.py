"""Pipeline and Step definitions for composing f3dasm blocks."""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
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
    name : str
        Human-readable name for this step (used in logs and SLURM
        job names).
    block : Block | DataGenerator | Callable
        The operation to execute. Can be a :class:`Block`,
        :class:`DataGenerator`, or a plain callable with signature
        ``(project_dir, project_job, **kwargs) -> None``.
    parallel : bool
        If ``True``, this step is executed as a SLURM array job
        (or with multiprocessing locally). Only meaningful when
        the block is a :class:`DataGenerator`.
    resources : SlurmResources
        SLURM resource requirements for this step.
    dependency : Literal["afterok", "afterany"]
        SLURM dependency type for the previous step.
        Must be ``"afterok"`` or ``"afterany"``.
    kwargs : dict[str, Any]
        Extra keyword arguments forwarded to the block's
        ``call`` method at execution time.
    """

    name: str = ""
    block: Block | DataGenerator | Callable | None = None
    parallel: bool = False
    resources: SlurmResources = field(default_factory=SlurmResources)
    dependency: Literal["afterok", "afterany"] = "afterok"
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

    Parameters
    ----------
    name : str
        Name of the pipeline (used for directory naming and SLURM
        job prefixes).
    steps : list[Step | Loop]
        The ordered sequence of steps and loops.

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
                scratch_dir="/scratch/user",
            ),
        )
    """

    name: str = ""
    steps: list[PipelineElement] = field(default_factory=list)

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
        mode: str = "local",
        cluster: SlurmCluster | None = None,
        project_dir: str | None = None,
        project_job: str | None = None,
    ) -> str:
        """Execute the pipeline.

        Parameters
        ----------
        mode : str
            Execution mode: ``"local"`` or ``"slurm"``.
        cluster : SlurmCluster, optional
            Cluster configuration (required when
            ``mode="slurm"``).
        project_dir : str, optional
            Override the project directory. Defaults to the current
            working directory (local) or the cluster scratch
            directory (SLURM).
        project_job : str, optional
            Existing project job ID for resumption. If ``None``,
            a new ID is generated.

        Returns
        -------
        str
            The project job ID.
        """
        # Lazy imports to avoid circular dependency:
        # pipeline -> executors -> pipeline
        from .executors.local import LocalExecutor
        from .executors.slurm import SlurmExecutor

        if mode == "local":
            executor: LocalExecutor | SlurmExecutor = LocalExecutor()
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
            project_dir=project_dir,
            project_job=project_job,
        )

    def generate_scripts(
        self,
        cluster: SlurmCluster,
        project_dir: str | None = None,
        project_job: str = "PLACEHOLDER",
        n_jobs: int = 1,
    ) -> dict[str, str]:
        """Generate SLURM scripts without submitting them.

        Useful for inspecting or manually editing scripts before
        submission.

        Parameters
        ----------
        cluster : SlurmCluster
            Cluster configuration.
        project_dir : str, optional
            Override the project directory.
        project_job : str
            Project job ID to embed in scripts.
        n_jobs : int
            Number of jobs for parallel steps.

        Returns
        -------
        dict[str, str]
            Mapping of ``"step_name"`` (or
            ``"step_name_loopN"``) to the rendered sbatch script
            content.
        """
        from .executors.slurm import SlurmExecutor

        executor = SlurmExecutor(cluster=cluster)
        return executor.generate_scripts(
            pipeline=self,
            project_dir=project_dir,
            project_job=project_job,
            n_jobs=n_jobs,
        )

    def resume(
        self,
        project_job: str,
        from_step: str | None = None,
        mode: str = "local",
        cluster: SlurmCluster | None = None,
        project_dir: str | None = None,
    ) -> str:
        """Resume a previously interrupted pipeline.

        Delegates to :meth:`run` with the given ``project_job``
        and optional ``from_step`` to skip completed phases.

        Parameters
        ----------
        project_job : str
            The project job ID to resume.
        from_step : str, optional
            Name of the step to resume from. If ``None``, resumes
            from the beginning.
        mode : str
            Execution mode: ``"local"`` or ``"slurm"``.
        cluster : SlurmCluster, optional
            Cluster configuration (required for ``"slurm"``).
        project_dir : str, optional
            Override the project directory.

        Returns
        -------
        str
            The project job ID.
        """
        return self.run(
            mode=mode,
            cluster=cluster,
            project_dir=project_dir,
            project_job=project_job,
            from_step=from_step,
        )
