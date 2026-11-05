# Wave-based array submission for parallel steps

## Status

accepted

## Context

A parallel step is submitted as one SLURM array whose width is capped by
`SlurmResources.max_array_size` (cluster policy). When the number of open
jobs exceeds that cap, each array task used to evaluate the strided slice
`open_jobs[k::max_array_size]` sequentially — so the `time`/`mem` a user
declares on `SlurmResources` silently had to cover `ceil(N / max_array_size)`
experiments, where `N` is not even knowable when the config is written (the
array width is resolved from `count_open` at submission time). Oversized
per-task budgets caused timeout censoring of long evaluations and poor
backfill scheduling.

## Decision

`SlurmResources` describes the cost of exactly `max_jobs_per_task`
experiments (default **1**), independent of `N`. When
`N > max_array_size × max_jobs_per_task`, the orchestrator submits the step
as multiple sequential **waves** (see `CONTEXT.md`): array submissions of at
most `max_array_size` tasks, each wave gated on the previous wave
*terminating* (`afterany`), tracked by a fourth orchestrator counter
(`WAVE_COUNT`) and threaded to workers via `F3DASM_WAVE` (the
`F3DASM_ITERATION` mechanism). Task `k` of wave `b` owns
`open_jobs[b·W·j:][k::W][:j]` — the direct generalization of the old stride
expression (`b=0`, `j=None` reproduces it exactly).

Wave assignment reads the **frozen snapshot** of open jobs in the central
store; per-sample result JSONs written during the step are deliberately not
consulted between waves. All tasks of all waves therefore see one immutable
assignment: race-free, deterministic wave count, trivially terminating, and
no mid-step mutation of the central store.

`max_jobs_per_task=None` restores the single strided submission and renders
byte-identical orchestrator/step scripts to the previous behaviour.

## Considered options

- **Consolidate-and-recount between waves** (merge sample JSONs into the
  central store, re-run `count_open`, take the first `min(remaining, W·j)`
  open jobs per wave). Rejected: it smuggles in retry-of-abandoned-jobs
  semantics that were explicitly not the goal, mutates the central store
  mid-step (changing crash/resume behaviour), needs a new consolidation
  entry point, and a wave that makes zero progress (e.g. partition outage)
  loops forever without an extra guard. If retry is ever wanted, it should
  be designed on top of the frozen-snapshot mechanism, not instead of it.
- **Chaining waves with the step's `dependency` (default `afterok`)**.
  Rejected: one infra-killed task (timeout, OOM, node death) would leave the
  next wake `DependencyNeverSatisfied` and orphan every remaining wave.
  Tighter per-task budgets make such kills *more* likely, and f3dasm's
  failure contract lives in per-sample job status, not SLURM exit codes.
  Consequence: the next pipeline step's `dependency` now attaches to the
  *final wave only* — earlier waves have already terminated (any state) by
  construction. Accumulating all wave ids into one `--dependency` list is
  not viable because SLURM purges completed-job records after `MinJobAge`.
- **Keeping striding as the default** (`max_jobs_per_task=None`). Rejected:
  a resource contract that only holds behind an opt-in flag is barely a
  contract. The default flip only changes behaviour in the overflow regime
  `N > max_array_size`, where the declared resources were already being
  violated, and the failure direction is safe (budgets sized for a strided
  slice over-provision a single experiment). Shipped as a minor release
  (2.3.0) with `max_jobs_per_task=None` as the one-line rollback.
