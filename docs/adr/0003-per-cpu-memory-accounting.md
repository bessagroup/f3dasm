# Per-CPU memory accounting in rendered sbatch scripts

## Status

accepted

## Context

Both script renderers (`render_sbatch_script`, `render_orchestrator_script`)
built a fixed `#SBATCH` header that always emitted per-node `--mem` and `--nodes`
and *never* emitted `--ntasks`. On any SLURM site whose `job_submit` filter
mandates a task count and per-CPU memory accounting — notably TU Delft's
DelftBlue — the very first `sbatch` (the orchestrator, and equally every step)
is rejected at submission time:

```
error: This job submission doesn't specify the number of tasks. Specify '--ntasks=<ntasks>'.
error: This job submission doesn't specify the amount of memory per-cpu (or per-gpu). Specify '--mem-per-cpu=<mem>'.
```

so a `SlurmCluster` pipeline could not run there at all — no partial degradation,
nothing submits. Crucially there was **no config-only workaround**:
`SlurmResources.extra_sbatch` can add `--mem-per-cpu`, but the hard-coded `--mem`
line is still emitted alongside it, and SLURM treats `--mem` together with
`--mem-per-cpu` as a **fatal** error (`--mem, --mem-per-cpu, and --mem-per-gpu
are mutually exclusive`), not a warning — on *every* site, not just DelftBlue.
This mutual exclusivity is what makes `extra_sbatch` structurally insufficient:
the fix has to be able to *omit* `--mem`, which only the renderer can do.

Cross-checked against Brown's Oscar (plain SLURM, no per-CPU mandate): the flags
emitted before this change (`--mem=8G --nodes=1`, no `--ntasks`) submit cleanly
there, and `--mem` + `--mem-per-cpu` together fail there too — confirming the
constraint is general SLURM behaviour, not a DelftBlue quirk, and that Oscar must
not regress.

## Decision

`SlurmResources` gains two fields, `ntasks: int = 1` and
`mem_per_cpu: str | None = None`, and both renderers share one
`_render_resource_directives(res)` helper (so the two headers cannot drift) that:

- **Emits `--ntasks={res.ntasks}` unconditionally, on every cluster.** This is a
  deliberate choice, not an oversight: `--ntasks=1` was confirmed harmless on
  Oscar (a plain-SLURM site) and a single always-present field is simpler than a
  second implicit on/off switch keyed off `mem_per_cpu`.
- **Chooses the memory directive by `mem_per_cpu`:** when it is set, emit
  `--mem-per-cpu={res.mem_per_cpu}` and **omit `--mem` entirely**; otherwise emit
  the per-node `--mem={res.mem}` (unchanged). When `mem_per_cpu` is set, `mem` is
  ignored.
- **Omits `--nodes` only when `mem_per_cpu` is set *and* `nodes == 1`** (its
  default): per-task allocation makes it redundant and it silences DelftBlue's
  benign "`--nodes` without `--exclusive`" warning. An explicit `nodes > 1` is
  always emitted — an explicit multi-node request is never silently dropped.

`mem`/`mem_per_cpu` precedence is **documented, not validated**: a dataclass
cannot distinguish "user left `mem` at its default" from "user set `mem` to that
same value," so any `__post_init__` "both set" guard would false-positive. Site
*values* (keeping `cpus_per_task × mem_per_cpu` under a partition's
`MaxMemPerCPU`) remain the caller's responsibility; f3dasm only makes the per-CPU
model *expressible*.

Backward compatibility: `mem_per_cpu` defaults to `None`, so per-node `--mem`
output is unchanged wherever it is not opted into. The only behaviour change for
existing configs is the added `--ntasks=1` line (confirmed harmless on both
DelftBlue and Oscar). All six golden fixtures under `tests/pipeline/golden/*.sh`
gain that one line and were regenerated accordingly.

### Consequence: the orchestrator's default resources

`_DEFAULT_ORCH_RESOURCES` (in `slurm.py`) keeps `mem="1G"` with
`mem_per_cpu=None`, so a DelftBlue pipeline that does **not** override
`Pipeline.orchestrator_resources` will still render `--mem=1G` for the
orchestrator job and be rejected. To run end-to-end on a per-CPU site the caller
must set `orchestrator_resources=SlurmResources(mem_per_cpu=..., ...)` as well as
per-step resources. This is intentional: baking a DelftBlue-specific value into
the framework default is exactly the site-specific concern this change leaves to
the caller — the default deliberately stays per-node.

## Considered options

- **Gating `--ntasks` behind `mem_per_cpu`.** Rejected: it makes the header
  depend on a second implicit switch for no benefit, since `--ntasks=1` is
  harmless where it is not required. One always-present field is simpler.
- **A `__post_init__` guard rejecting "both `mem` and `mem_per_cpu` set."**
  Rejected: dataclass defaults make "explicitly set to the default" and "left at
  the default" indistinguishable, so the check would false-positive on
  legitimate configs. Precedence is documented instead.
- **Fixing it in `extra_sbatch` / leaving the header alone.** Rejected: `--mem`
  and `--mem-per-cpu` are mutually exclusive (fatal) at every SLURM site, so the
  hard-coded `--mem` must be *omittable* — a config appended *after* the fixed
  header cannot achieve that.
- **Validating `mem_per_cpu` against a partition's `MaxMemPerCPU`.** Rejected as
  out of scope: f3dasm cannot know per-partition caps, and this stays a
  site/caller concern (mirrors the resource-value stance of ADR 0002).
