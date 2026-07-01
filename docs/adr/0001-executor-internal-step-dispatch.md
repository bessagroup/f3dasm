# Executor-internal step dispatch via a serializable execution context

## Status

accepted

## Context

Running one pipeline step — load `ExperimentData`, `arm` the block, branch on block
type (`DataGenerator` / `Block` / callable), pick a parallelization mode, and persist
the result — was implemented twice: once in `LocalExecutor._run_step_locally` and once
in the SLURM worker's `_execute_step`. The two are near-identical and must be edited in
lockstep, so they drift. The branches also disagreed in small ways (e.g. `result.store()`
vs `result.store(run_dir)`), and `DataGenerator.call(mode=...)` carries an implicit,
mode-dependent return-vs-persist contract that each executor re-derived — including a
latent gap where `parallel_mode="parallel"` returns data that the local path never stored.

## Decision

Introduce one internal dispatcher, `run_step(step, run_dir, ctx)`, that both executors
call. It owns the whole skeleton — the `from_file → arm` preamble, the block-type ladder,
the array striding, and a single uniform persistence rule: `if result is not None:
result.store(run_dir)` (the self-persisting cluster modes return `None` and are untouched).
This also fixes the `parallel_mode="parallel"` persistence gap.

The per-environment difference is carried by an **execution context** — a small,
*serializable value* (parallelization mode, this invocation's job index, array size) with
one `execute(block, data)` method. The local loop and the SLURM worker each construct one.

The public seam `DataGenerator.call(mode=...)` is **not** changed; this work is confined to
executor-internal code.

## Considered options

- **Abstract hooks on the `Executor` base** (each subclass overrides `run_datagenerator`,
  etc.). Rejected: the SLURM worker is a *separate process* that unpickles only the
  `Pipeline`/`Step` — it never has the `SlurmExecutor` instance, so a behavioural hook
  object isn't available there. The variation must cross a cloudpickle/CLI boundary, which
  forces a **value**, not a strategy object. This is the non-obvious constraint; do not
  "simplify" the execution-context value back into methods on `Executor`.
- **Redesigning `DataGenerator.call` to drop the `mode` string.** Rejected for now:
  `DataGenerator` is public API; that would be a breaking change requiring a deprecation
  path. Kept out of scope.
- **Preserving the per-mode persistence special-casing bug-for-bug.** Rejected: it carries
  the `parallel_mode="parallel"` non-persist gap forward into the freshly-deepened module.
