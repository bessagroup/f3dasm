"""SLURM executor — submits pipeline steps as SLURM jobs."""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Third-party
import cloudpickle

# Local
from ..loop import Loop
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

# Default resources for the orchestrator job (lightweight — it only
# runs ``sbatch`` commands and exits).
_DEFAULT_ORCH_RESOURCES = SlurmResources(
    time="00:10:00", mem="1G", cpus_per_task=1, nodes=1
)


@dataclass
class SlurmExecutor(Executor):
    """Execute a pipeline by submitting SLURM jobs.

    A single self-resubmitting **orchestrator** script manages the
    entire pipeline. It uses a ``STEP_COUNT`` (which pipeline
    element to handle) and a ``LOOP_COUNT`` (current loop
    iteration) to progress through the pipeline one step or loop
    iteration at a time.

    At submission time the submitter's ``sys.path`` is stored as
    ``.sys_path.json`` alongside ``.pipeline.pkl``. When a SLURM
    job unpickles the pipeline, these paths are restored so that
    imports from local scripts resolve correctly. This requires
    compute nodes to share a filesystem with the submission host.

    Parameters
    ----------
    cluster : SlurmCluster
        Cluster-specific configuration.
    """

    cluster: SlurmCluster

    def run(
        self,
        pipeline: Pipeline,
        project_job: str | None = None,
        rootdir: Path | None = None,
    ) -> str:
        """Submit the pipeline to SLURM.

        Generates bash scripts for every pipeline element, renders
        a single orchestrator script, and submits it via
        ``sbatch``. The orchestrator handles all step submissions
        and dependency chaining.

        The current ``sys.path`` is normalized (resolved to
        absolute paths, empty strings expanded to ``cwd``) and
        stored as ``.sys_path.json`` in the job directory. SLURM
        jobs restore these paths before deserializing the pipeline
        so that imports from local scripts work on compute nodes.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline to execute.
        project_job : str, optional
            Job identifier used as the run folder
            (``rootdir / project_job``). If ``None``, a
            timestamp-based ID is generated.
        rootdir : Path, optional
            Root directory under which the job folder is created.
            Defaults to the current working directory.

        Returns
        -------
        str
            The project job ID.
        """
        rootdir = rootdir if rootdir is not None else Path.cwd()
        resolved_job: str = project_job or str(int(time.time()))

        # job_dir holds all pipeline artifacts (.pipeline.pkl,
        # slurm_scripts/, logs/). ExperimentData for each step
        # lives in job_dir / step.project_dir.
        job_dir: Path = rootdir / resolved_job
        job_dir.mkdir(parents=True, exist_ok=True)

        pipeline_path: Path = job_dir / ".pipeline.pkl"
        with open(pipeline_path, "wb") as f:
            cloudpickle.dump(pipeline, f)
        logger.info(f"Pipeline serialized to {pipeline_path}")

        # Store the submitter's sys.path so that SLURM jobs can
        # resolve imports from local scripts (e.g. from my_script
        # import func). Paths are normalized to absolute to avoid
        # ambiguity when the job runs in a different cwd.
        resolved_paths: list[str] = []
        for p in sys.path:
            canonical = (
                str(Path(p).resolve()) if p else str(Path.cwd().resolve())
            )
            if canonical not in resolved_paths:
                resolved_paths.append(canonical)

        sys_path_path: Path = job_dir / ".sys_path.json"
        with open(sys_path_path, "w") as f:
            json.dump(resolved_paths, f)
        logger.info(f"sys.path serialized to {sys_path_path}")

        # Create the log and script directories.
        log_dir: Path = job_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        script_dir: Path = job_dir / "slurm_scripts"
        script_dir.mkdir(parents=True, exist_ok=True)

        # --- Generate step scripts for all pipeline elements ---
        script_paths: dict[str, str] = {}
        for i, element in enumerate(pipeline.steps):
            if isinstance(element, Step):
                label = element.name
                script = render_sbatch_script(
                    step=element,
                    cluster=self.cluster,
                    pipeline_name=pipeline.name,
                    label=label,
                    job_dir=job_dir,
                    n_jobs=element.array_jobs,
                    iteration=0,
                )
                path = script_dir / f"{label}.sh"
                path.write_text(script)
                script_paths[label] = str(path)

            elif isinstance(element, Loop):
                for step in element.steps:
                    label = f"loop{i}_{step.name}"
                    script = render_sbatch_script(
                        step=step,
                        cluster=self.cluster,
                        pipeline_name=pipeline.name,
                        label=label,
                        job_dir=job_dir,
                        n_jobs=step.array_jobs,
                        iteration="$F3DASM_ITERATION",
                    )
                    path = script_dir / f"{label}.sh"
                    path.write_text(script)
                    script_paths[label] = str(path)

        # --- Generate and write the orchestrator ---
        orch_res = pipeline.orchestrator_resources or _DEFAULT_ORCH_RESOURCES
        orch_script = render_orchestrator_script(
            pipeline=pipeline,
            cluster=self.cluster,
            orchestrator_resources=orch_res,
            script_paths=script_paths,
            log_dir_path=str(log_dir),
        )
        orch_path = script_dir / "orchestrator.sh"
        orch_path.write_text(orch_script)

        # --- Submit the orchestrator ---
        cmd: list[str] = ["sbatch", str(orch_path), "0", "0"]
        logger.info(f"Submitting orchestrator: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True
        )
        job_id = result.stdout.strip().split()[-1]
        logger.info(f"  -> SLURM orchestrator job {job_id}")

        return resolved_job

    def generate_scripts(
        self,
        pipeline: Pipeline,
        project_job: str = "PLACEHOLDER",
        rootdir: Path | None = None,
    ) -> dict[str, str]:
        """Generate SLURM scripts without submitting.

        Returns all step scripts and the orchestrator script. For
        loop body steps, ``$F3DASM_ITERATION`` is used as the
        iteration placeholder.

        Parameters
        ----------
        pipeline : Pipeline
            The pipeline to generate scripts for.
        project_job : str
            Placeholder project job ID.
        rootdir : Path, optional
            Root directory under which the job folder is created.
            Defaults to the current working directory.

        Returns
        -------
        dict[str, str]
            Mapping of label to rendered script content.
        """
        rootdir = rootdir if rootdir is not None else Path.cwd()
        job_dir: Path = rootdir / project_job
        log_dir_path: str = str(job_dir / "logs")

        scripts: dict[str, str] = {}
        # Placeholder paths for the orchestrator (since scripts
        # are not written to disk in generate_scripts)
        placeholder_paths: dict[str, str] = {}

        for i, element in enumerate(pipeline.steps):
            if isinstance(element, Step):
                label = element.name
                scripts[label] = render_sbatch_script(
                    step=element,
                    cluster=self.cluster,
                    pipeline_name=pipeline.name,
                    label=label,
                    job_dir=job_dir,
                    n_jobs=element.array_jobs,
                    iteration=0,
                )
                placeholder_paths[label] = f"SCRIPT_DIR/{label}.sh"

            elif isinstance(element, Loop):
                for step in element.steps:
                    label = f"loop{i}_{step.name}"
                    scripts[label] = render_sbatch_script(
                        step=step,
                        cluster=self.cluster,
                        pipeline_name=pipeline.name,
                        label=label,
                        job_dir=job_dir,
                        n_jobs=step.array_jobs,
                        iteration="$F3DASM_ITERATION",
                    )
                    placeholder_paths[label] = f"SCRIPT_DIR/{label}.sh"

        orch_res = pipeline.orchestrator_resources or _DEFAULT_ORCH_RESOURCES
        scripts["orchestrator"] = render_orchestrator_script(
            pipeline=pipeline,
            cluster=self.cluster,
            orchestrator_resources=orch_res,
            script_paths=placeholder_paths,
            log_dir_path=log_dir_path,
        )

        return scripts


#                                                              Script rendering
# =============================================================================


def render_sbatch_script(
    step: Step,
    cluster: SlurmCluster,
    pipeline_name: str,
    label: str,
    job_dir: Path,
    n_jobs: int | None,
    iteration: int | str,
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
    job_dir : Path
        Absolute path to the job directory (``rootdir/project_job``).
        Pipeline artifacts and per-step ExperimentData live here.
    n_jobs : int | None
        Number of jobs for array steps.
    iteration : int | str
        Current loop iteration index. Can be a shell variable
        reference (e.g. ``"$F3DASM_ITERATION"``) for scripts
        used inside an orchestrator loop.

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
    log_path: str = str(job_dir / "logs" / label)
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
        f"  --job-dir={job_dir}",
        f"  --project-dir={step.project_dir}",
        f"  --iteration={iteration}",
    ]

    if step.parallel:
        cmd_parts.append("  --job-number=$SLURM_ARRAY_TASK_ID")

    lines.append(" \\\n".join(cmd_parts))
    lines.append("")

    return "\n".join(lines)


def render_orchestrator_script(
    pipeline: Pipeline,
    cluster: SlurmCluster,
    orchestrator_resources: SlurmResources,
    script_paths: dict[str, str],
    log_dir_path: str,
) -> str:
    """Render a self-resubmitting orchestrator for the pipeline.

    The orchestrator manages the entire pipeline using two
    counters passed as positional arguments:

    - ``STEP_COUNT``: index into the pipeline's top-level
      elements (Steps and Loops).
    - ``LOOP_COUNT``: current iteration within a Loop (0 when
      not inside a loop).

    Each execution handles exactly one action (one Step
    submission or one Loop iteration), then resubmits itself
    with ``--dependency`` on the last submitted job. The
    dependency type is determined by the *next* step's
    ``Step.dependency`` field.

    Parameters
    ----------
    pipeline : Pipeline
        The full pipeline definition.
    cluster : SlurmCluster
        Cluster configuration.
    orchestrator_resources : SlurmResources
        SLURM resources for the orchestrator job.
    script_paths : dict[str, str]
        Mapping of label to absolute script path on disk.
    log_dir_path : str
        Directory for orchestrator log files.

    Returns
    -------
    str
        The rendered orchestrator bash script.
    """
    res = orchestrator_resources
    total_steps = len(pipeline.steps)

    # --- SBATCH header ---
    lines: list[str] = [
        "#!/bin/bash",
        f"#SBATCH --job-name=orchestrator_{pipeline.name}",
        f"#SBATCH --time={res.time}",
        f"#SBATCH --mem={res.mem}",
        f"#SBATCH --cpus-per-task={res.cpus_per_task}",
        f"#SBATCH --nodes={res.nodes}",
        f"#SBATCH --partition={cluster.partition}",
        f"#SBATCH --account={cluster.account}",
        f"#SBATCH --output={log_dir_path}/orchestrator_%j.out",
    ]

    for key, val in res.extra_sbatch.items():
        lines.append(f"#SBATCH --{key}={val}")

    lines.extend(
        [
            "",
            "STEP_COUNT=$1",
            "LOOP_COUNT=$2",
            'SELF=$(realpath "$0")',
            f"TOTAL_STEPS={total_steps}",
            "",
            'while [ "$STEP_COUNT" -lt "$TOTAL_STEPS" ]; do',
            "",
        ]
    )

    # --- Generate if/elif blocks for each pipeline element ---
    for i, element in enumerate(pipeline.steps):
        # Determine the condition keyword
        cond = "if" if i == 0 else "elif"
        lines.append(f'  {cond} [ "$STEP_COUNT" -eq {i} ]; then')

        if isinstance(element, Step):
            _render_step_block(
                lines=lines,
                step=element,
                step_index=i,
                pipeline=pipeline,
                script_paths=script_paths,
                total_steps=total_steps,
            )

        elif isinstance(element, Loop):
            _render_loop_block(
                lines=lines,
                loop=element,
                step_index=i,
                pipeline=pipeline,
                script_paths=script_paths,
                total_steps=total_steps,
            )

        lines.append("")

    lines.extend(
        [
            "  fi",
            "done",
            "",
            'echo "Pipeline complete."',
            "",
        ]
    )

    return "\n".join(lines)


def _get_next_dependency(
    pipeline: Pipeline,
    current_index: int,
    total_steps: int,
) -> str | None:
    """Get the dependency type for the next element after current_index.

    Returns the ``Step.dependency`` of the next element (or the
    first inner step of a Loop). Returns ``None`` if there is no
    next element.
    """
    next_idx = current_index + 1
    if next_idx >= total_steps:
        return None

    next_element = pipeline.steps[next_idx]
    if isinstance(next_element, Step):
        return next_element.dependency
    elif isinstance(next_element, Loop):
        if next_element.steps:
            return next_element.steps[0].dependency
    return "afterok"


def _render_step_block(
    lines: list[str],
    step: Step,
    step_index: int,
    pipeline: Pipeline,
    script_paths: dict[str, str],
    total_steps: int,
) -> None:
    """Append bash lines for a Step element in the orchestrator."""
    label = step.name
    script_path = script_paths[label]

    lines.extend(
        [
            f"    # Step: {step.name}",
            f'    RESULT=$(sbatch "{script_path}")',
            "    JOB_ID=$(echo $RESULT | awk '{print $NF}')",
            f'    echo "Submitted {step.name}: job $JOB_ID"',
        ]
    )

    next_step = step_index + 1
    next_dep = _get_next_dependency(pipeline, step_index, total_steps)

    if next_dep is not None:
        lines.extend(
            [
                f"    STEP_COUNT={next_step}",
                f"    sbatch --dependency={next_dep}:$JOB_ID"
                f' "$SELF" $STEP_COUNT $LOOP_COUNT',
                "    exit 0",
            ]
        )
    else:
        # Last element in pipeline — don't resubmit
        lines.append("    exit 0")


def _render_loop_block(
    lines: list[str],
    loop: Loop,
    step_index: int,
    pipeline: Pipeline,
    script_paths: dict[str, str],
    total_steps: int,
) -> None:
    """Append bash lines for a Loop element in the orchestrator."""
    n_iters = loop.n_iterations

    lines.extend(
        [
            f"    # Loop: {n_iters} iterations",
            f'    if [ "$LOOP_COUNT" -lt {n_iters} ]; then',
            "      export F3DASM_ITERATION=$LOOP_COUNT",
        ]
    )

    # Submit each inner step with dependency chaining
    for j, inner_step in enumerate(loop.steps):
        inner_label = f"loop{step_index}_{inner_step.name}"
        inner_path = script_paths[inner_label]

        if j == 0:
            # First inner step: no dependency
            lines.extend(
                [
                    f"      # Inner step: {inner_step.name}",
                    f'      RESULT=$(sbatch --export=ALL "{inner_path}")',
                    "      PREV_JOB_ID=$(echo $RESULT | awk '{print $NF}')",
                    f'      echo "  Submitted {inner_step.name}'
                    f' (iter $LOOP_COUNT): job $PREV_JOB_ID"',
                ]
            )
        else:
            dep = inner_step.dependency
            lines.extend(
                [
                    f"      # Inner step: {inner_step.name}",
                    f"      RESULT=$(sbatch"
                    f" --dependency={dep}:$PREV_JOB_ID"
                    f' --export=ALL "{inner_path}")',
                    "      PREV_JOB_ID=$(echo $RESULT | awk '{print $NF}')",
                    f'      echo "  Submitted {inner_step.name}'
                    f' (iter $LOOP_COUNT): job $PREV_JOB_ID"',
                ]
            )

    # Resubmit orchestrator for next iteration
    # Use the first inner step's dependency type for iteration
    # resubmission (determines if next iteration runs on failure)
    iter_dep = loop.steps[0].dependency if loop.steps else "afterok"

    lines.extend(
        [
            "      LOOP_COUNT=$((LOOP_COUNT + 1))",
            f"      sbatch --dependency={iter_dep}:$PREV_JOB_ID"
            f' "$SELF" $STEP_COUNT $LOOP_COUNT',
            "      exit 0",
            "    else",
            "      # Loop done — advance to next element",
            "      LOOP_COUNT=0",
            f"      STEP_COUNT={step_index + 1}",
            "      continue",
            "    fi",
        ]
    )
