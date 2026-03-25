"""SLURM executor — submits pipeline steps as SLURM jobs."""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

# Third-party
import cloudpickle

# Local
from ..pipeline import Pipeline, Step
from ..resources import SlurmCluster, SlurmResources
from .base import Executor

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Stable"
# =============================================================================
#
# =============================================================================


logger = logging.getLogger("f3dasm")


@dataclass
class SlurmExecutor(Executor):
    """Execute a pipeline by submitting SLURM jobs.

    Each :class:`Step` becomes one ``sbatch`` submission.
    Dependencies between consecutive steps are expressed via
    ``--dependency``.  Parallel steps are submitted as array
    jobs.

    Parameters
    ----------
    cluster : SlurmCluster
        Cluster-specific configuration.
    """

    cluster: SlurmCluster

    def run(
        self,
        pipeline: Pipeline,
        project_dir: str | Path | None = None,
        project_job: str | None = None,
    ) -> str:
        """Submit the pipeline to SLURM.

        Iterates over the flattened pipeline steps, renders an
        sbatch script for each, writes it to disk, and submits
        with ``sbatch``. Consecutive steps are chained via SLURM
        job dependencies (``--dependency=afterok:<id>``).

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline to execute.
        project_dir : str | Path, optional
            Base project directory. Defaults to the cluster's
            scratch directory.
        project_job : str, optional
            Project job ID for resumption. If ``None``, the first
            submitted SLURM job ID is used.

        Returns
        -------
        str
            The project job ID.
        """
        resolved_dir: Path = (
            Path(project_dir or self.cluster.scratch_dir) / pipeline.name
        )

        # Resolve the project_job upfront so it can be embedded in
        # scripts and used as the pickle path before any job is
        # submitted. When resuming, the caller provides the existing
        # job ID; otherwise we generate one from the current time.
        resolved_job: str = project_job or str(int(time.time()))

        # Create the run directory and serialize the pipeline so
        # each SLURM node can deserialize it via run_step.py.
        run_dir: Path = resolved_dir / resolved_job
        run_dir.mkdir(parents=True, exist_ok=True)

        pipeline_path: Path = run_dir / ".pipeline.pkl"
        with open(pipeline_path, "wb") as f:
            cloudpickle.dump(pipeline, f)
        logger.info(f"Pipeline serialized to {pipeline_path}")

        # Create the log directory for SLURM output files.
        log_dir: Path = resolved_dir / self.cluster.log_dir.format(
            project_job=resolved_job
        )
        log_dir.mkdir(parents=True, exist_ok=True)

        # Flatten the pipeline into linear step sequence
        flat_steps: list[tuple[Step, int, int]] = pipeline._flatten()
        prev_job_id: str | None = None

        for step, iteration, n_iterations in flat_steps:
            # Build a unique label for this submission
            # (e.g. "run_loop2" for iteration 2 of a loop)
            label: str = step.name
            if n_iterations > 1:
                label = f"{step.name}_loop{iteration}"

            # Render the sbatch script for this step
            script: str = render_sbatch_script(
                step=step,
                cluster=self.cluster,
                pipeline_name=pipeline.name,
                label=label,
                project_dir=resolved_dir,
                project_job=resolved_job,
                n_jobs=step.array_jobs,
                # TODO n_jobs needs to represent the number of experiments
                iteration=iteration,
            )

            # Write script to disk for auditability
            script_dir: Path = run_dir / "slurm_scripts"
            script_dir.mkdir(parents=True, exist_ok=True)
            script_path: Path = (script_dir / f"{label}").with_suffix(".sh")
            script_path.write_text(script)

            # Build and run the sbatch command with dependency
            # chaining to the previous step
            cmd: list[str] = ["sbatch"]
            if prev_job_id is not None:
                cmd.append(f"--dependency={step.dependency}:{prev_job_id}")
            cmd.append(str(script_path))

            logger.info(f"Submitting step {label!r}: {' '.join(cmd)}")
            result: subprocess.CompletedProcess[str] = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )

            # Parse the SLURM job ID from sbatch stdout
            # Expected format: "Submitted batch job 12345678"
            job_id: str = result.stdout.strip().split()[-1]
            prev_job_id = job_id

            logger.info(
                f"  -> SLURM job {job_id} ({step.name}, iter={iteration})"
            )

        return resolved_job

    def generate_scripts(
        self,
        pipeline: Pipeline,
        project_dir: str | Path | None = None,
        project_job: str = "PLACEHOLDER",
        n_jobs: int = 1,
    ) -> dict[str, str]:
        """Generate SLURM scripts without submitting.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline to generate scripts for.
        project_dir : str | Path, optional
            Base project directory.
        project_job : str
            Placeholder project job ID.
        n_jobs : int
            Number of jobs for parallel steps.

        Returns
        -------
        dict[str, str]
            Mapping of label to rendered script content.
        """
        resolved_dir: Path = (
            Path(project_dir)
            if project_dir
            else Path(self.cluster.scratch_dir)
        ) / pipeline.name

        scripts: dict[str, str] = {}
        flat_steps: list[tuple[Step, int, int]] = pipeline._flatten()

        for step, iteration, n_iterations in flat_steps:
            label: str = step.name
            if n_iterations > 1:
                label = f"{step.name}_loop{iteration}"

            scripts[label] = render_sbatch_script(
                step=step,
                cluster=self.cluster,
                pipeline_name=pipeline.name,
                label=label,
                project_dir=resolved_dir,
                project_job=project_job,
                n_jobs=n_jobs,
                iteration=iteration,
            )

        return scripts


#                                                              Script rendering
# =============================================================================


def render_sbatch_script(
    step: Step,
    cluster: SlurmCluster,
    pipeline_name: str,
    label: str,
    project_dir: Path,
    project_job: str,
    n_jobs: int | None,
    iteration: int,
) -> str:
    """Render a complete sbatch script for a single step.

    This is a pure function: it takes all the information it
    needs as arguments and returns the script as a string.
    The generated script invokes ``f3dasm._src.pipeline.run_step``
    as its payload.

    Parameters
    ----------
    step : Step
        The pipeline step to render.
    cluster : SlurmCluster
        Cluster configuration (partition, account, etc.).
    pipeline_name : str
        Name of the pipeline (used in job names).
    label : str
        Unique label for this submission (used in filenames).
    project_dir : Path
        Base project directory.
    project_job : str
        Project job identifier.
    n_jobs : int | None
        Number of jobs for array steps.
    iteration : int
        Current loop iteration index.

    Returns
    -------
    str
        The rendered sbatch script content.
    """
    res: SlurmResources = step.resources

    # --- SBATCH header ---
    lines: list[str] = [
        "#!/bin/bash",
        f"#SBATCH --job-name={label}_{pipeline_name}",
        f"#SBATCH --time={res.time}",
        f"#SBATCH --mem={res.mem}",
        f"#SBATCH --cpus-per-task={res.cpus_per_task}",
        f"#SBATCH --nodes={res.nodes}",
        f"#SBATCH --partition={cluster.partition}",
        f"#SBATCH --account={cluster.account}",
    ]

    # --- Array job configuration (parallel steps only) ---
    if step.parallel and n_jobs is not None:
        array_size: int = min(n_jobs, res.max_array_size)
        lines.append(f"#SBATCH --array=0-{array_size}%{res.max_concurrent}")

    # --- Log output paths ---
    log_dir: str = cluster.log_dir.format(project_job=project_job)
    log_path: str = str(project_dir / log_dir / label)
    if step.parallel:
        lines.append(f"#SBATCH --output={log_path}_%A_%a.out")
    else:
        lines.append(f"#SBATCH --output={log_path}_%j.out")

    # --- Extra user-specified sbatch directives ---
    for key, val in res.extra_sbatch.items():
        lines.append(f"#SBATCH --{key}={val}")

    lines.append("")

    # --- Cluster-specific environment setup ---
    # (module loads, library path fixes, etc.)
    for cmd in cluster.env_setup:
        lines.append(cmd)
    if cluster.env_setup:
        lines.append("")

    # --- Environment variables ---
    for key, val in cluster.env_vars.items():
        lines.append(f'export {key}="{val}"')

    # Pin thread counts to SLURM allocation
    lines.append("export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK")
    lines.append("export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK")
    lines.append("")

    # --- Python command ---
    # Invoke the run_step CLI entry point with all necessary
    # context for this step
    run_step_module: str = "f3dasm._src.pipeline.run_step"
    cmd_parts: list[str] = [
        f"{cluster.runner} -m {run_step_module}",
        f"  --step={step.name}",
        f"  --project-dir={project_dir}",
        f"  --project-job={project_job}",
        f"  --pipeline-name={pipeline_name}",
        f"  --iteration={iteration}",
    ]

    if step.parallel:
        cmd_parts.append("  --job-number=$SLURM_ARRAY_TASK_ID")

    lines.append(" \\\n".join(cmd_parts))
    lines.append("")

    return "\n".join(lines)
