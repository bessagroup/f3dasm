# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository

`f3dasm` is a Python framework for data-driven design and analysis of structures and materials. It's a published package (PyPI / conda-forge, JOSS paper) — public API stability matters. Supports Python ≥3.10.

## Common commands

The project is managed with `uv`. Use `uv run <cmd>` to run inside the project environment.

- Run full test suite: `uv run pytest` (will fail if coverage drops below 85%)
- Run smoke tests only: `uv run pytest -m smoke` (or `make test`)
- Run a single test file / test: `uv run pytest tests/pipeline/test_pipeline.py::test_name -v`
- Skip tests requiring a specific optional dependency: `uv run pytest -S optuna` (or `-S all` to skip every dep-gated test)
- Lint: `make lint` (i.e. `ruff check`). Pre-commit also runs `ruff-format` and `ruff-check --fix`.
- Build docs locally: `mkdocs build`
- Build distribution: `make build` (`uv build`)

CI (`.github/workflows/pull_request.yml`) runs `pytest` (with `[tests,all]` extras) on the matrix of Python 3.10–3.13 × {ubuntu, windows, macos}, plus `mkdocs build` and `pre-commit run --all-files`. Mirror that locally before pushing.

## Architecture

The user-facing package surface lives in `src/f3dasm/__init__.py`; all real code is under `src/f3dasm/_src/` (the `_src` prefix marks it as the private implementation — re-export new public symbols from the top-level `__init__.py` and `__all__`). The submodule `f3dasm.design` etc. are thin re-export shims at `src/f3dasm/<name>.py`.

The framework revolves around four abstractions that compose together. Understanding their relationship is essential before changing core behavior:

1. **`Domain`** (`_src/design/domain.py`) — declares the input/output parameter space (continuous/int/categorical/constant/array). Used by samplers and the `@datagenerator` decorator to wire inputs/outputs.
2. **`ExperimentData`** (`_src/experimentdata.py`) — the central data container; a table of `ExperimentSample` rows holding inputs, outputs, and per-sample status. Persistable to disk; this is the value type that flows between blocks.
3. **`Block`** (`_src/core.py`) — the unit of computation. Every operation (sampling, evaluation, optimizer update step) is a `Block` with the uniform signature `call(data: ExperimentData, **kwargs) -> ExperimentData`. There is no separate `Sampler` or `Optimizer` class hierarchy — that was deliberately collapsed; do not reintroduce one. Optional one-time setup goes in `arm(data)` (e.g. fitting a surrogate). Composition operators:
   - `>>` returns a `ChainedBlock` running blocks in order.
   - `.loop(n)` returns a `LoopBlock` repeating the block n times, feeding each iteration's output into the next.
4. **`Pipeline`** (`_src/pipeline/`) — chains `Step`s (and `Loop`s) into a runnable workflow. Executes locally or on SLURM via `SlurmCluster` / `SlurmResources`; pipelines are resumable. Local Python imports (`from my_script import func`) are preserved on cluster compute nodes via shared filesystem.

`DataGenerator` (`_src/core.py`) is a special non-Block base class for per-sample evaluators. Its `call` method dispatches across parallelization modes (`sequential`, `parallel`, `cluster`, `mpi`, `cluster_array`) defined in `_src/datagenerator.py`. The `@datagenerator(output_names=[...], domain=...)` decorator wraps a plain Python function into a `DataGenerator` instance.

Factories (`create_sampler`, `create_datagenerator`, `create_optimizer`) live in `_src/samplers.py`, `_src/datageneration/datagenerator_factory.py`, and `_src/optimization/optimizer_factory.py`. `create_optimizer` returns just the **update step** Block for ask/tell optimizers — the caller owns the loop. One-shot scipy optimizers (`cg`, `lbfgsb`, `nelder_mead`) drive their own inner loop.

Hydra integration: `Block.from_yaml(init_config, call_config)` instantiates blocks from `DictConfig` so workflows can be configured externally.

Optional deps (`optuna`, `abaqus2py`) are gated behind the `all` extra and behind the `requires_dependency(name)` pytest marker — check `_src/optimization/_imports.py` for the lazy-import pattern when adding new optional integrations.

## Conventions

- `ruff` line length is **79**. The lint config selects `E,W,F,I,B,UP` and ignores `E226,E3,E731,C901,UP045,B027,B012`. Tests/docs/studies/notebooks are excluded from lint.
- Docstrings are NumPy-style and used by `mkdocstrings`. Match the existing style of `core.py` when adding new public API. The custom docstring agent at `.github/agents/docstring_agent.agent.md` enforces these rules and is forbidden from changing code logic — keep that contract if invoked.
- Public API additions must be re-exported from `src/f3dasm/__init__.py` (and added to `__all__`) and covered in `tests/test_public_imports.py`.
- Tests use the `smoke` marker for the fast subset; mark tests that need optional deps with `@pytest.mark.requires_dependency("optuna")` etc.
- Do not commit changes to `coverage_html_report/`, `dist/`, `build/`, or `site/` — these are generated.

## Related repositories

`f3dasm` is the data-driven framework underpinning the L2CO ecosystem (Bessa Research Group). Related repositories:

- [l2co](https://github.com/bessagroup/L2CO) — Learning to Choose Optimizers: a meta-learner that selects an optimizer from problem features before any evaluations, then reassesses that choice from the observed optimization trajectory.
- [rl2co](https://github.com/bessagroup/rl2co) — Reinforcement Learning to Choose Optimizers: a JAX-based RL agent that dynamically switches between optimizers during a run.
- [l2co-tasks](https://github.com/bessagroup/l2co-tasks) — Optimization task definitions (BBOB, CEC 2005, PDE, spiral, …) compatible with the L2CO library.
- [l2co_experiments](https://github.com/bessagroup/l2co_experiments) — Hydra + f3dasm experiment pipelines (dataset creation, training, rollouts, figures) for the L2CO studies.
- [agentic-l2co](https://github.com/bessagroup/agentic-l2co) — An LLM-agent drop-in replacement for `l2co.L2COModel`, driving two-stage optimizer selection with an Ollama-hosted LLM.
- [bbob-jax](https://github.com/bessagroup/bbob-jax) — JAX implementations of the BBOB and CEC 2005 black-box optimization benchmark functions.
- [f3dasm](https://github.com/bessagroup/f3dasm) — Framework for Data-Driven Design and Analysis of Structures and Materials; provides `ExperimentData`, pipelines, and SLURM orchestration.

## Agent skills

### Issue tracker

Issues and PRDs live in the [bessagroup/f3dasm](https://github.com/bessagroup/f3dasm) GitHub Issues (managed via the `gh` CLI). External PRs are **not** a triage surface. See `docs/agents/issue-tracker.md`.

### Triage labels

Canonical role names used verbatim — `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, plus the repo's existing `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
