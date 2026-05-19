# Config + Budget + Ollama Backend — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if
> subagents available) or superpowers:executing-plans to implement this plan. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-study `config.yaml` as the single config source, enforce a
wall-clock time budget (`budget: "hh:mm:ss"`), and implement an Ollama backend so the
runtime can use locally-running open-source models.

**Architecture:** `StudyConfig` is loaded from `<study-dir>/config.yaml` and merged
with CLI overrides before `AgenticRun` is constructed. `AgenticRun` gains `_budget` /
`_remaining()` and injects `remaining_time` into every `Task` and `Report`. A new
`backends/ollama.py` implements `StrategizerSession` and `ImplementerSession` using
Ollama's Python client with a manual multi-turn tool-calling loop.

**Tech Stack:** Python stdlib (`dataclasses`, `datetime.timedelta`, `subprocess`),
PyYAML (already a transitive dep — confirm before adding), `ollama` Python package
(optional, lazy import).

---

## Chunk 1: StudyConfig + config.yaml loading

### Task 1: `StudyConfig` dataclass and `_load_study_config`

**Files:**
- Modify: `src/f3dasm/_src/agentic/agent_runtime.py` (after `Report` dataclass, ~line 139)
- Test: `tests/agentic/test_agent_runtime.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/agentic/test_agent_runtime.py`:

```python
import textwrap
from datetime import timedelta


# ── StudyConfig ──────────────────────────────────────────────────────────────

def test_load_study_config_missing_file(tmp_path):
    """Absent config.yaml → all defaults."""
    from f3dasm._src.agentic.agent_runtime import _load_study_config
    cfg = _load_study_config(tmp_path)
    assert cfg.model is None
    assert cfg.backend == "claude"
    assert cfg.budget is None
    assert cfg.checkpoint_every is None


def test_load_study_config_full(tmp_path):
    """All keys parsed correctly."""
    from f3dasm._src.agentic.agent_runtime import _load_study_config
    (tmp_path / "config.yaml").write_text(textwrap.dedent("""\
        model: llama3.1:8b
        backend: ollama
        budget: "01:30:00"
        checkpoint_every: 10
    """))
    cfg = _load_study_config(tmp_path)
    assert cfg.model == "llama3.1:8b"
    assert cfg.backend == "ollama"
    assert cfg.budget == timedelta(hours=1, minutes=30)
    assert cfg.checkpoint_every == 10


def test_load_study_config_budget_only(tmp_path):
    """Partial config — only budget set."""
    from f3dasm._src.agentic.agent_runtime import _load_study_config
    (tmp_path / "config.yaml").write_text('budget: "00:45:00"\n')
    cfg = _load_study_config(tmp_path)
    assert cfg.budget == timedelta(minutes=45)
    assert cfg.model is None


def test_load_study_config_unknown_key_raises(tmp_path):
    """Unrecognised key → AgenticRunError."""
    from f3dasm._src.agentic.agent_runtime import (
        AgenticRunError,
        _load_study_config,
    )
    (tmp_path / "config.yaml").write_text("unknown_key: value\n")
    with pytest.raises(AgenticRunError, match="unknown_key"):
        _load_study_config(tmp_path)


def test_load_study_config_bad_budget_raises(tmp_path):
    """Malformed budget string → AgenticRunError."""
    from f3dasm._src.agentic.agent_runtime import (
        AgenticRunError,
        _load_study_config,
    )
    (tmp_path / "config.yaml").write_text('budget: "not-a-time"\n')
    with pytest.raises(AgenticRunError, match="budget"):
        _load_study_config(tmp_path)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/agentic/test_agent_runtime.py \
    -k "study_config" -v
```

Expected: `ImportError` or `AttributeError` (symbols not yet defined).

- [ ] **Step 3: Implement `StudyConfig` and `_load_study_config`**

In `agent_runtime.py`, add after the `Report` dataclass and before the
`AgenticRunError` class:

```python
# ---------------------------------------------------------------------------
# Study-level config
# ---------------------------------------------------------------------------

from datetime import timedelta   # add to top-of-file imports


_KNOWN_CONFIG_KEYS: frozenset[str] = frozenset(
    {"model", "backend", "budget", "checkpoint_every"}
)


def _parse_budget(value: str) -> timedelta:
    """Parse an hh:mm:ss string into a timedelta.

    Parameters
    ----------
    value : str
        String of the form ``"HH:MM:SS"``.

    Returns
    -------
    timedelta

    Raises
    ------
    ValueError
        If *value* does not match the expected format.
    """
    parts = value.strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"expected HH:MM:SS, got {value!r}")
    h, m, s = (int(p) for p in parts)
    return timedelta(hours=h, minutes=m, seconds=s)


@dataclass
class StudyConfig:
    """Runtime configuration for one agentic study.

    All fields default to ``None``, meaning "use the built-in default".
    The CLI and ``AgenticRun.__init__`` apply further resolution.

    Parameters
    ----------
    model : str or None
        LLM model identifier.
    backend : str
        Backend name — ``"claude"`` or ``"ollama"``.
    budget : timedelta or None
        Wall-clock time limit; ``None`` means unbounded.
    checkpoint_every : int or None
        Implementer-call cadence override; ``None`` uses
        :data:`CHECKPOINT_EVERY`.
    """

    model: str | None = None
    backend: str = "claude"
    budget: timedelta | None = None
    checkpoint_every: int | None = None


def _load_study_config(study_dir: Path) -> StudyConfig:
    """Load ``<study_dir>/config.yaml`` into a :class:`StudyConfig`.

    Parameters
    ----------
    study_dir : Path
        Root of the study tree.

    Returns
    -------
    StudyConfig
        All fields are ``None`` / defaults if the file is absent.

    Raises
    ------
    AgenticRunError
        On unknown keys or malformed ``budget`` values.
    """
    import yaml  # soft import — only needed at load time

    config_path = Path(study_dir) / "config.yaml"
    if not config_path.exists():
        return StudyConfig()

    raw = yaml.safe_load(config_path.read_text()) or {}
    unknown = set(raw) - _KNOWN_CONFIG_KEYS
    if unknown:
        raise AgenticRunError(
            f"config.yaml: unknown key(s): {', '.join(sorted(unknown))}. "
            f"Valid keys: {', '.join(sorted(_KNOWN_CONFIG_KEYS))}."
        )

    budget: timedelta | None = None
    if "budget" in raw:
        try:
            budget = _parse_budget(str(raw["budget"]))
        except (ValueError, TypeError) as exc:
            raise AgenticRunError(
                f"config.yaml: invalid budget {raw['budget']!r} — "
                f"expected HH:MM:SS ({exc})."
            ) from exc

    return StudyConfig(
        model=raw.get("model"),
        backend=raw.get("backend", "claude"),
        budget=budget,
        checkpoint_every=raw.get("checkpoint_every"),
    )
```

Also add `timedelta` to the `from datetime import datetime, timezone` line at the top.

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/agentic/test_agent_runtime.py \
    -k "study_config" -v
```

Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/f3dasm/_src/agentic/agent_runtime.py \
        tests/agentic/test_agent_runtime.py
git commit -m "feat(agentic): add StudyConfig dataclass and _load_study_config"
```

---

### Task 2: Wire `StudyConfig` into `AgenticRun.__init__` and the CLI

**Files:**
- Modify: `src/f3dasm/_src/agentic/agent_runtime.py` (`AgenticRun.__init__`)
- Modify: `src/f3dasm/agentic/__main__.py`
- Modify: `src/f3dasm/agentic/__init__.py`
- Test: `tests/agentic/test_agent_runtime.py`

- [ ] **Step 1: Write failing tests**

```python
def test_agentic_run_accepts_study_config(tmp_path):
    """AgenticRun accepts a StudyConfig and uses its checkpoint_every."""
    from f3dasm._src.agentic.agent_runtime import (
        AgenticRun,
        StudyConfig,
    )
    (tmp_path / "PROBLEM_STATEMENT.md").write_text("# Test\n")
    cfg = StudyConfig(checkpoint_every=5)
    run = AgenticRun(
        tmp_path,
        study_config=cfg,
        strategizer_factory=lambda **kw: _make_stub_strategizer("Done", ""),
        implementer_factory=lambda **kw: _make_stub_implementer(),
    )
    assert run._checkpoint_every == 5


def test_agentic_run_cli_overrides_config(tmp_path):
    """Explicit CLI model kwarg overrides config.yaml model."""
    from f3dasm._src.agentic.agent_runtime import (
        AgenticRun,
        StudyConfig,
    )
    (tmp_path / "PROBLEM_STATEMENT.md").write_text("# Test\n")
    cfg = StudyConfig(model="model-from-config")
    run = AgenticRun(
        tmp_path,
        model="model-from-cli",  # CLI wins
        study_config=cfg,
        strategizer_factory=lambda **kw: _make_stub_strategizer("Done", ""),
        implementer_factory=lambda **kw: _make_stub_implementer(),
    )
    assert run._model == "model-from-cli"
```

(Reuse or add `_make_stub_strategizer` / `_make_stub_implementer` helpers
already used in the test file.)

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/agentic/test_agent_runtime.py \
    -k "accepts_study_config or cli_overrides" -v
```

Expected: `TypeError` (unexpected keyword `study_config`).

- [ ] **Step 3: Add `study_config` parameter to `AgenticRun.__init__`**

In `AgenticRun.__init__`, add:

```python
study_config: StudyConfig | None = None,
```

Then in the body, after `self._study_dir = ...`:

```python
# Merge study config → CLI overrides.
_cfg = study_config or StudyConfig()
# model: explicit kwarg wins over config, config wins over backend default.
if model is None:
    model = _cfg.model  # may still be None → resolved below
self._model: str = (
    model if model is not None else self._backend.default_model
)
# checkpoint_every: explicit kwarg wins over config.
if checkpoint_every == CHECKPOINT_EVERY and _cfg.checkpoint_every is not None:
    checkpoint_every = _cfg.checkpoint_every
self._checkpoint_every = checkpoint_every
# budget (new field).
self._budget: timedelta | None = _cfg.budget
self._start_time: datetime | None = None  # set in execute()
```

Remove the existing `self._model` and `self._checkpoint_every` assignments that are
now handled above.

- [ ] **Step 4: Update the CLI (`__main__.py`)**

Replace `_build_parser` and `main` with:

```python
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m f3dasm.agentic",
        description=(
            "Run the agentic-f3dasm v2 runtime against a study directory."
        ),
    )
    parser.add_argument(
        "study_dir",
        metavar="study-dir",
        type=Path,
        help="Path to the study directory. Must contain PROBLEM_STATEMENT.md.",
    )
    parser.add_argument(
        "--model",
        default=None,
        metavar="MODEL",
        help="LLM model identifier (overrides config.yaml).",
    )
    parser.add_argument(
        "--backend",
        default=None,
        choices=["claude", "ollama"],
        help="Backend to use (overrides config.yaml; default: claude).",
    )
    parser.add_argument(
        "--budget",
        default=None,
        metavar="HH:MM:SS",
        help="Wall-clock time budget (overrides config.yaml).",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        metavar="N",
        dest="checkpoint_every",
        help="Delegations between checkpoints (overrides config.yaml).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    from .._src.agentic.agent_runtime import (
        CHECKPOINT_EVERY,
        AgenticRun,
        AgenticRunError,
        StudyConfig,
        _load_study_config,
        _parse_budget,
    )
    from .._src.agentic.backends.claude import CLAUDE_BACKEND

    parser = _build_parser()
    args = parser.parse_args(argv)
    study_dir = Path(args.study_dir)

    # Load file config, then apply CLI overrides.
    cfg = _load_study_config(study_dir)
    if args.model is not None:
        cfg = StudyConfig(
            model=args.model,
            backend=cfg.backend,
            budget=cfg.budget,
            checkpoint_every=cfg.checkpoint_every,
        )
    if args.backend is not None:
        cfg = StudyConfig(
            model=cfg.model,
            backend=args.backend,
            budget=cfg.budget,
            checkpoint_every=cfg.checkpoint_every,
        )
    if args.budget is not None:
        try:
            parsed_budget = _parse_budget(args.budget)
        except ValueError as exc:
            print(f"Error: invalid --budget: {exc}", file=sys.stderr)
            return 1
        cfg = StudyConfig(
            model=cfg.model,
            backend=cfg.backend,
            budget=parsed_budget,
            checkpoint_every=cfg.checkpoint_every,
        )
    if args.checkpoint_every is not None:
        cfg = StudyConfig(
            model=cfg.model,
            backend=cfg.backend,
            budget=cfg.budget,
            checkpoint_every=args.checkpoint_every,
        )

    # Resolve backend.
    if cfg.backend == "ollama":
        from .._src.agentic.backends.ollama import OLLAMA_BACKEND
        backend = OLLAMA_BACKEND
    else:
        backend = CLAUDE_BACKEND

    run = AgenticRun(
        study_dir=study_dir,
        study_config=cfg,
        backend=backend,
    )

    try:
        deliverable_path = run.execute()
    except AgenticRunError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(str(deliverable_path))
    return 0
```

- [ ] **Step 5: Export `StudyConfig` from `f3dasm.agentic`**

In `src/f3dasm/agentic/__init__.py`, add `StudyConfig` to the imports and `__all__`:

```python
from .._src.agentic.agent_runtime import (
    CHECKPOINT_EVERY,
    MVP_DEFAULT_MODEL,
    AgenticRun,
    AgenticRunError,
    Report,
    StudyConfig,   # new
    Task,
)
```

- [ ] **Step 6: Run all agentic tests**

```bash
uv run pytest tests/agentic/ -v
```

Expected: all existing tests plus new ones PASS.

- [ ] **Step 7: Commit**

```bash
git add src/f3dasm/_src/agentic/agent_runtime.py \
        src/f3dasm/agentic/__init__.py \
        src/f3dasm/agentic/__main__.py \
        tests/agentic/test_agent_runtime.py
git commit -m "feat(agentic): wire StudyConfig into AgenticRun and CLI"
```

---

## Chunk 2: Time Budget Enforcement

### Task 3: `_remaining()` and pre-delegation budget check

**Files:**
- Modify: `src/f3dasm/_src/agentic/agent_runtime.py`
- Test: `tests/agentic/test_agent_runtime.py`

- [ ] **Step 1: Write failing tests**

```python
from datetime import timedelta
import time


def test_remaining_no_budget(tmp_path):
    """_remaining() returns None when no budget is set."""
    from f3dasm._src.agentic.agent_runtime import AgenticRun, StudyConfig
    (tmp_path / "PROBLEM_STATEMENT.md").write_text("# t\n")
    run = AgenticRun(
        tmp_path,
        study_config=StudyConfig(),
        strategizer_factory=lambda **kw: _make_stub_strategizer("Done", ""),
        implementer_factory=lambda **kw: _make_stub_implementer(),
    )
    run._start_time = datetime.now(tz=timezone.utc)
    assert run._remaining() is None


def test_remaining_with_budget(tmp_path):
    """_remaining() returns a positive timedelta just after start."""
    from f3dasm._src.agentic.agent_runtime import AgenticRun, StudyConfig
    (tmp_path / "PROBLEM_STATEMENT.md").write_text("# t\n")
    run = AgenticRun(
        tmp_path,
        study_config=StudyConfig(budget=timedelta(hours=1)),
        strategizer_factory=lambda **kw: _make_stub_strategizer("Done", ""),
        implementer_factory=lambda **kw: _make_stub_implementer(),
    )
    run._start_time = datetime.now(tz=timezone.utc)
    rem = run._remaining()
    assert rem is not None
    assert timedelta(minutes=59) < rem <= timedelta(hours=1)


def test_budget_expired_skips_delegation(tmp_path):
    """When budget is exhausted, _tool_delegate triggers clean shutdown."""
    from f3dasm._src.agentic.agent_runtime import AgenticRun, StudyConfig
    (tmp_path / "PROBLEM_STATEMENT.md").write_text("# t\n")
    # Budget already expired (negative timedelta via a past start_time).
    run = AgenticRun(
        tmp_path,
        study_config=StudyConfig(budget=timedelta(seconds=1)),
        strategizer_factory=lambda **kw: _make_stub_strategizer("Done", ""),
        implementer_factory=lambda **kw: _make_stub_implementer(),
    )
    run._start_time = datetime.now(tz=timezone.utc) - timedelta(hours=1)
    # Manually call _tool_delegate; should set _done_called and return
    # without contacting the Implementer.
    run._implementer = _make_stub_implementer()
    result = run._tool_delegate(
        intent="do something", expected_report="report"
    )
    assert run._done_called
    assert "budget" in result.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/agentic/test_agent_runtime.py \
    -k "remaining or budget_expired" -v
```

Expected: `AttributeError` (`_remaining` not defined).

- [ ] **Step 3: Implement `_remaining()` and budget check in `execute()` and `_tool_delegate()`**

In `AgenticRun`, add after `__init__`:

```python
def _remaining(self) -> timedelta | None:
    """Return wall-clock time left, or None if no budget is set.

    Returns
    -------
    timedelta or None
        Remaining time.  Can be negative (budget exceeded).
    """
    if self._budget is None or self._start_time is None:
        return None
    return self._budget - (
        datetime.now(tz=timezone.utc) - self._start_time
    )
```

In `execute()`, after `self._init_git()` and before `briefing_text = ...`:

```python
self._start_time = datetime.now(tz=timezone.utc)
```

At the **top** of `_tool_delegate`, before the git commit block, add:

```python
# Budget check — must happen before any work for this delegation.
remaining = self._remaining()
if remaining is not None and remaining <= timedelta(0):
    self._done_called = True
    self._logger.info(
        "Budget exhausted before delegation #%d — stopping cleanly.",
        self._total_delegations,
    )
    return (
        "BUDGET_EXHAUSTED: wall-clock budget has been consumed. "
        "No further delegations will be issued."
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/agentic/test_agent_runtime.py \
    -k "remaining or budget_expired" -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/f3dasm/_src/agentic/agent_runtime.py \
        tests/agentic/test_agent_runtime.py
git commit -m "feat(agentic): implement _remaining() and pre-delegation budget check"
```

---

### Task 4: Inject `remaining_time` into `Task` and `Report`; update `_format_task`; update deliverable metadata

**Files:**
- Modify: `src/f3dasm/_src/agentic/agent_runtime.py`
- Test: `tests/agentic/test_agent_runtime.py`

- [ ] **Step 1: Write failing tests**

```python
def test_format_task_no_budget():
    """No remaining_time → no time line in formatted output."""
    from f3dasm._src.agentic.agent_runtime import Task, _format_task
    task = Task(intent="do X", expected_report="report Y")
    rendered = _format_task(task)
    assert "Time remaining" not in rendered


def test_format_task_with_budget():
    """remaining_time present → time line appears in formatted output."""
    from f3dasm._src.agentic.agent_runtime import Task, _format_task
    task = Task(
        intent="do X",
        expected_report="report Y",
        remaining_time=timedelta(hours=1, minutes=12, seconds=47),
    )
    rendered = _format_task(task)
    assert "Time remaining: 01:12:47" in rendered


def test_format_task_budget_warning():
    """remaining_time < 20% of budget → warning line in formatted output."""
    from f3dasm._src.agentic.agent_runtime import Task, _format_task
    task = Task(
        intent="do X",
        expected_report="report Y",
        remaining_time=timedelta(minutes=5),
        budget=timedelta(hours=1),   # 5/60 ≈ 8% < 20%
    )
    rendered = _format_task(task)
    assert "nearly exhausted" in rendered


def test_solution_md_includes_budget_metadata(tmp_path):
    """solution.md contains budget and time_used when budget is set."""
    # Run a minimal stub run with a budget.
    from f3dasm._src.agentic.agent_runtime import AgenticRun, StudyConfig
    (tmp_path / "PROBLEM_STATEMENT.md").write_text("# t\n")
    cfg = StudyConfig(budget=timedelta(hours=1))
    run = AgenticRun(
        tmp_path,
        study_config=cfg,
        strategizer_factory=lambda **kw: _make_stub_strategizer(
            "Done", "great result"
        ),
        implementer_factory=lambda **kw: _make_stub_implementer(),
        stdin=StringIO(""),
        stdout=StringIO(),
    )
    deliv = run.execute()
    solution = (deliv / "solution.md").read_text()
    assert "budget: 1:00:00" in solution
    assert "time_used:" in solution
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/agentic/test_agent_runtime.py \
    -k "format_task or solution_md_includes_budget" -v
```

Expected: failures on `remaining_time` field missing and missing metadata.

- [ ] **Step 3: Add fields to `Task` and `Report`; update `_format_task`**

Update `Task`:

```python
@dataclass
class Task:
    """..."""
    intent: str
    expected_report: str
    remaining_time: timedelta | None = None
    budget: timedelta | None = None   # needed for the 20% warning
```

Update `Report`:

```python
@dataclass
class Report:
    """..."""
    actions_taken: str
    files_touched: list[str]
    conclusions: str
    numbers: dict[str, Any]
    raw: str
    remaining_time: timedelta | None = None
```

Replace `_format_task`:

```python
def _format_task(task: Task) -> str:
    """Render a Task as a structured markdown block."""
    lines = [
        "## Task",
        "",
        "### Intent",
        task.intent,
        "",
        "### Expected report",
        task.expected_report,
    ]
    if task.remaining_time is not None:
        h, rem = divmod(int(task.remaining_time.total_seconds()), 3600)
        m, s = divmod(rem, 60)
        lines.insert(2, f"**Time remaining: {h:02d}:{m:02d}:{s:02d}**")
        lines.insert(3, "")
        if (
            task.budget is not None
            and task.remaining_time < 0.2 * task.budget
        ):
            lines.insert(4, (
                "⚠ Budget nearly exhausted — "
                "scope remaining work accordingly."
            ))
            lines.insert(5, "")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Inject `remaining_time` when constructing `Task` in `_tool_delegate`**

Replace the `Task(...)` construction in `_tool_delegate`:

```python
remaining = self._remaining()
task = Task(
    intent=intent,
    expected_report=expected_report,
    remaining_time=remaining,
    budget=self._budget,
)
```

Inject into Report before returning to Strategizer (at the success path, after
`self._last_report = report`):

```python
report.remaining_time = self._remaining()
```

- [ ] **Step 5: Add budget/time_used to `_assemble_deliverable`**

In `solution_lines`, after `f"- run_dir: {self._run_dir}"`, add:

```python
if self._budget is not None:
    solution_lines.append(f"- budget: {self._budget}")
if self._start_time is not None:
    used = datetime.now(tz=timezone.utc) - self._start_time
    h, rem = divmod(int(used.total_seconds()), 3600)
    m, s = divmod(rem, 60)
    solution_lines.append(f"- time_used: {h:02d}:{m:02d}:{s:02d}")
```

- [ ] **Step 6: Run all agentic tests**

```bash
uv run pytest tests/agentic/ -v
```

Expected: all tests PASS.

- [ ] **Step 7: Commit**

```bash
git add src/f3dasm/_src/agentic/agent_runtime.py \
        tests/agentic/test_agent_runtime.py
git commit -m "feat(agentic): inject remaining_time into Task/Report; budget metadata in solution.md"
```

---

## Chunk 3: Ollama Backend

### Task 5: `IMPLEMENTER_SYSTEM_PROMPT_OLLAMA` in `agent_prompts.py`

**Files:**
- Modify: `src/f3dasm/_src/agentic/agent_prompts.py`
- Test: `tests/agentic/test_agent_prompts.py`

- [ ] **Step 1: Write a minimal test**

```python
def test_ollama_implementer_prompt_exists():
    from f3dasm._src.agentic.agent_prompts import (
        IMPLEMENTER_SYSTEM_PROMPT_OLLAMA,
    )
    assert "bash" in IMPLEMENTER_SYSTEM_PROMPT_OLLAMA.lower()
    assert "## Report" in IMPLEMENTER_SYSTEM_PROMPT_OLLAMA
```

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run pytest tests/agentic/test_agent_prompts.py \
    -k "ollama_implementer_prompt" -v
```

Expected: `ImportError`.

- [ ] **Step 3: Add `IMPLEMENTER_SYSTEM_PROMPT_OLLAMA` to `agent_prompts.py`**

Append to `agent_prompts.py`:

```python
IMPLEMENTER_SYSTEM_PROMPT_OLLAMA: str = """\
You are the Implementer in an agentic research system. You receive a Task from
the Strategizer and must complete it using a single tool: **bash**.

## Your only tool: bash

Use `bash` for everything: reading files, writing files, running Python
scripts, installing nothing (assume the environment is fixed). All work must
stay inside the study directory you were given at the start.

Examples:
- Read a file:   bash(cmd="cat workspace/results.csv")
- Write a file:  bash(cmd="python3 -c \\"open('workspace/out.py','w').write('...')\\"")
- Run a script:  bash(cmd="python3 workspace/optimise.py")

## Your output

When you have finished the task, emit **exactly** the following block and
nothing else after it:

## Report

### Actions taken
- <bullet per action>

### Files touched
- <path>

### Conclusions
<free-form prose — what you found, what worked, what failed, any anomaly>

### Numbers
<key>: <value>

Never omit a section. Use "- none" if a section is empty. Every anomaly,
error, or unexpected result belongs in ### Conclusions.
"""
```

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run pytest tests/agentic/test_agent_prompts.py \
    -k "ollama_implementer_prompt" -v
```

- [ ] **Step 5: Commit**

```bash
git add src/f3dasm/_src/agentic/agent_prompts.py \
        tests/agentic/test_agent_prompts.py
git commit -m "feat(agentic): add IMPLEMENTER_SYSTEM_PROMPT_OLLAMA"
```

---

### Task 6: `backends/ollama.py` — core implementation

**Files:**
- Create: `src/f3dasm/_src/agentic/backends/ollama.py`
- Test: `tests/agentic/test_ollama_backend.py` (new file)

- [ ] **Step 1: Write failing tests** (create `tests/agentic/test_ollama_backend.py`)

```python
"""Tests for the Ollama backend.

All tests mock the ``ollama`` package so no Ollama server is required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ollama_response(content: str, tool_calls=None):
    """Build a minimal mock ollama response."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls or []
    resp = MagicMock()
    resp.message = msg
    return resp


# ---------------------------------------------------------------------------
# _preflight_ollama
# ---------------------------------------------------------------------------

def test_preflight_ollama_server_unreachable():
    """Raises AgenticRunError when Ollama server is down."""
    from f3dasm._src.agentic.backends.ollama import _preflight_ollama
    from f3dasm._src.agentic.agent_runtime import AgenticRunError

    with patch(
        "f3dasm._src.agentic.backends.ollama.ollama"
    ) as mock_ollama:
        mock_ollama.list.side_effect = Exception("connection refused")
        with pytest.raises(AgenticRunError, match="Ollama server"):
            _preflight_ollama("qwen2.5:0.5b")


def test_preflight_ollama_model_missing():
    """Raises AgenticRunError when requested model is not pulled."""
    from f3dasm._src.agentic.backends.ollama import _preflight_ollama
    from f3dasm._src.agentic.agent_runtime import AgenticRunError

    with patch(
        "f3dasm._src.agentic.backends.ollama.ollama"
    ) as mock_ollama:
        model_entry = MagicMock()
        model_entry.model = "llama3.1:8b"
        mock_ollama.list.return_value = MagicMock(models=[model_entry])
        with pytest.raises(AgenticRunError, match="not found locally"):
            _preflight_ollama("qwen2.5:0.5b")


def test_preflight_ollama_ok():
    """No exception when server is up and model is available."""
    from f3dasm._src.agentic.backends.ollama import _preflight_ollama

    with patch(
        "f3dasm._src.agentic.backends.ollama.ollama"
    ) as mock_ollama:
        model_entry = MagicMock()
        model_entry.model = "qwen2.5:0.5b"
        mock_ollama.list.return_value = MagicMock(models=[model_entry])
        _preflight_ollama("qwen2.5:0.5b")  # must not raise


# ---------------------------------------------------------------------------
# _OllamaStrategizer
# ---------------------------------------------------------------------------

def test_ollama_strategizer_plain_text_return():
    """A response with no tool calls is returned as plain text."""
    from f3dasm._src.agentic.backends.ollama import _OllamaStrategizer

    with patch(
        "f3dasm._src.agentic.backends.ollama.ollama"
    ) as mock_ollama:
        mock_ollama.chat.return_value = _make_ollama_response(
            "Hello from Strategizer"
        )
        session = _OllamaStrategizer(
            system_prompt="sys",
            model="qwen2.5:0.5b",
            tool_closures={"Ask": lambda question: "ok"},
        )
        result = session.send("hi")
        assert result == "Hello from Strategizer"


def test_ollama_strategizer_tool_call_ask():
    """A tool call to Ask is forwarded to the closure and result fed back."""
    from f3dasm._src.agentic.backends.ollama import _OllamaStrategizer

    call = MagicMock()
    call.function.name = "Ask"
    call.function.arguments = {"question": "what is x?"}

    with patch(
        "f3dasm._src.agentic.backends.ollama.ollama"
    ) as mock_ollama:
        # First response: tool call; second: plain text (loop termination).
        mock_ollama.chat.side_effect = [
            _make_ollama_response("", tool_calls=[call]),
            _make_ollama_response("Done asking"),
        ]
        ask_results = []
        session = _OllamaStrategizer(
            system_prompt="sys",
            model="qwen2.5:0.5b",
            tool_closures={
                "Ask": lambda question: ask_results.append(question) or "user answer",
                "Delegate": lambda **kw: "delegate result",
                "Done": lambda summary: "done",
            },
        )
        result = session.send("start")
        assert "what is x?" in ask_results
        assert result == "Done asking"


# ---------------------------------------------------------------------------
# _OllamaImplementer
# ---------------------------------------------------------------------------

def test_ollama_implementer_bash_tool_executed(tmp_path):
    """bash tool call is executed via subprocess in study_dir."""
    from f3dasm._src.agentic.backends.ollama import _OllamaImplementer

    call = MagicMock()
    call.function.name = "bash"
    call.function.arguments = {"cmd": "echo hello"}

    with patch(
        "f3dasm._src.agentic.backends.ollama.ollama"
    ) as mock_ollama:
        mock_ollama.chat.side_effect = [
            _make_ollama_response("", tool_calls=[call]),
            _make_ollama_response("## Report\n### Actions taken\n- ran echo\n"
                                  "### Files touched\n- none\n"
                                  "### Conclusions\nok\n### Numbers\n"),
        ]
        session = _OllamaImplementer(
            system_prompt="sys",
            model="qwen2.5:0.5b",
            study_dir=tmp_path,
        )
        result = session.send("do task")
        assert "## Report" in result


def test_ollama_implementer_bash_output_in_history(tmp_path):
    """bash stdout is appended to message history before next call."""
    from f3dasm._src.agentic.backends.ollama import _OllamaImplementer

    call = MagicMock()
    call.function.name = "bash"
    call.function.arguments = {"cmd": "echo captured_output"}

    captured_histories = []

    def _mock_chat(model, messages, tools):
        captured_histories.append([m.copy() for m in messages])
        if len(captured_histories) == 1:
            return _make_ollama_response("", tool_calls=[call])
        return _make_ollama_response(
            "## Report\n### Actions taken\n- done\n"
            "### Files touched\n- none\n"
            "### Conclusions\nok\n### Numbers\n"
        )

    with patch(
        "f3dasm._src.agentic.backends.ollama.ollama"
    ) as mock_ollama:
        mock_ollama.chat.side_effect = _mock_chat
        session = _OllamaImplementer(
            system_prompt="sys",
            model="qwen2.5:0.5b",
            study_dir=tmp_path,
        )
        session.send("do task")
        # Second call's history should contain the bash output.
        second_call_messages = captured_histories[1]
        all_content = " ".join(
            str(m.get("content", "")) for m in second_call_messages
        )
        assert "captured_output" in all_content
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
uv run pytest tests/agentic/test_ollama_backend.py -v
```

Expected: `ModuleNotFoundError` (file doesn't exist yet).

- [ ] **Step 3: Implement `backends/ollama.py`**

Create `src/f3dasm/_src/agentic/backends/ollama.py`:

```python
"""Ollama backend for agentic-f3dasm.

Provides ``OLLAMA_BACKEND`` — a :class:`~backends.base.Backend` that
drives Strategizer and Implementer sessions using a locally-running
Ollama server instead of the Claude Agent SDK.

The Implementer's tool surface is a single ``bash`` tool executed via
``subprocess``.  The Strategizer's tools (Ask, Delegate, Done) are
semantic: the runtime intercepts them, not the backend.

The ``ollama`` Python package is imported lazily so that the rest of
agentic-f3dasm can be imported without it installed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from .base import Backend

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Elvis Aguero (elvis_alexander_aguero_vera@brown.edu)"
__credits__ = ["Elvis Aguero"]
__status__ = "Experimental"
# =============================================================================

OLLAMA_DEFAULT_MODEL: str = "qwen2.5:0.5b"

# Lazy module-level reference; populated on first use.
ollama: Any = None


def _import_ollama() -> None:
    global ollama
    if ollama is not None:
        return
    try:
        import ollama as _ollama
        ollama = _ollama
    except ImportError as exc:
        from ..agent_runtime import AgenticRunError
        raise AgenticRunError(
            "The 'ollama' Python package is required for the Ollama backend. "
            "Install it with: pip install ollama"
        ) from exc


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def _preflight_ollama(model: str) -> None:
    """Check that Ollama is running and *model* is available locally.

    Raises
    ------
    AgenticRunError
        If the server is unreachable or the model is not pulled.
    """
    from ..agent_runtime import AgenticRunError
    _import_ollama()
    try:
        response = ollama.list()
    except Exception as exc:
        raise AgenticRunError(
            f"Ollama server is not reachable: {exc}. "
            "Start it with: ollama serve"
        ) from exc
    available = [m.model for m in response.models]
    if model not in available:
        raise AgenticRunError(
            f"Model {model!r} not found locally. "
            f"Pull it with: ollama pull {model}\n"
            f"Available models: {available}"
        )


# ---------------------------------------------------------------------------
# Tool schema helpers
# ---------------------------------------------------------------------------

def _tool_schema(name: str, description: str, properties: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
            },
        },
    }


_STRATEGIZER_TOOLS = [
    _tool_schema("Ask", "Ask the user a clarifying question.",
                 {"question": {"type": "string"}}),
    _tool_schema("Delegate",
                 "Delegate a task to the Implementer.",
                 {"intent": {"type": "string"},
                  "expected_report": {"type": "string"}}),
    _tool_schema("Done",
                 "Signal that the run is complete.",
                 {"summary": {"type": "string"}}),
    _tool_schema("Read",
                 "Read a file from the study directory.",
                 {"path": {"type": "string"}}),
    _tool_schema("WriteMarkdown",
                 "Write a markdown note to strategizer_notes/.",
                 {"path": {"type": "string"},
                  "content": {"type": "string"}}),
]

_IMPLEMENTER_TOOLS = [
    _tool_schema("bash",
                 "Execute a shell command in the study directory.",
                 {"cmd": {"type": "string"}}),
]


# ---------------------------------------------------------------------------
# Session implementations
# ---------------------------------------------------------------------------

class _OllamaStrategizer:
    """Strategizer session backed by Ollama."""

    def __init__(
        self,
        *,
        system_prompt: str,
        model: str,
        tool_closures: dict,
    ) -> None:
        _import_ollama()
        self._model = model
        self._closures = tool_closures
        self._history: list[dict] = [
            {"role": "system", "content": system_prompt}
        ]

    def send(self, message: str) -> str:
        """Send *message* and run the tool loop until a text reply."""
        self._history.append({"role": "user", "content": message})
        while True:
            response = ollama.chat(
                model=self._model,
                messages=self._history,
                tools=_STRATEGIZER_TOOLS,
            )
            msg = response.message
            self._history.append(
                {"role": "assistant", "content": msg.content or "",
                 "tool_calls": [
                     {"function": {"name": tc.function.name,
                                   "arguments": tc.function.arguments}}
                     for tc in (msg.tool_calls or [])
                 ]}
            )
            if not msg.tool_calls:
                return msg.content or ""
            for call in msg.tool_calls:
                name = call.function.name
                args = call.function.arguments or {}
                fn = self._closures.get(name)
                result = fn(**args) if fn else f"ERROR: unknown tool {name}"
                self._history.append({
                    "role": "tool",
                    "content": str(result),
                })


class _OllamaImplementer:
    """Implementer session backed by Ollama with a single bash tool."""

    def __init__(
        self,
        *,
        system_prompt: str,
        model: str,
        study_dir: Path,
    ) -> None:
        _import_ollama()
        self._model = model
        self._study_dir = Path(study_dir)
        self._history: list[dict] = [
            {"role": "system", "content": system_prompt}
        ]

    def _run_bash(self, cmd: str) -> str:
        """Execute *cmd* in the study directory; return stdout+stderr."""
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=self._study_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = result.stdout
        if result.stderr:
            output += "\n[stderr]\n" + result.stderr
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return output or "(no output)"

    def send(self, message: str) -> str:
        """Send *message* and run the bash tool loop until a Report."""
        self._history.append({"role": "user", "content": message})
        while True:
            response = ollama.chat(
                model=self._model,
                messages=self._history,
                tools=_IMPLEMENTER_TOOLS,
            )
            msg = response.message
            self._history.append(
                {"role": "assistant", "content": msg.content or "",
                 "tool_calls": [
                     {"function": {"name": tc.function.name,
                                   "arguments": tc.function.arguments}}
                     for tc in (msg.tool_calls or [])
                 ]}
            )
            if not msg.tool_calls:
                return msg.content or ""
            for call in msg.tool_calls:
                if call.function.name == "bash":
                    output = self._run_bash(
                        call.function.arguments.get("cmd", "")
                    )
                else:
                    output = (
                        f"ERROR: unknown tool {call.function.name!r}. "
                        "Use 'bash' only."
                    )
                self._history.append({
                    "role": "tool",
                    "content": output,
                })


# ---------------------------------------------------------------------------
# Backend factory functions
# ---------------------------------------------------------------------------

def _make_ollama_strategizer(
    *, system_prompt: str, model: str, tool_closures: dict
) -> _OllamaStrategizer:
    return _OllamaStrategizer(
        system_prompt=system_prompt,
        model=model,
        tool_closures=tool_closures,
    )


def _make_ollama_implementer(
    *, system_prompt: str, model: str, study_dir: Path
) -> _OllamaImplementer:
    return _OllamaImplementer(
        system_prompt=system_prompt,
        model=model,
        study_dir=study_dir,
    )


# ---------------------------------------------------------------------------
# Public backend instance
# ---------------------------------------------------------------------------

OLLAMA_BACKEND: Backend = Backend(
    name="ollama",
    default_model=OLLAMA_DEFAULT_MODEL,
    strategizer_factory=_make_ollama_strategizer,
    implementer_factory=_make_ollama_implementer,
    preflight=lambda: None,   # preflight requires model; called with model
                               # in AgenticRun._preflight — see Note below.
)
```

> **Note on preflight:** `Backend.preflight` is a zero-arg callable per
> `base.py`. Model-aware preflight is called explicitly: update
> `AgenticRun._preflight` to pass `self._model` when the backend is ollama:
>
> ```python
> def _preflight(self) -> None:
>     ...
>     if self._backend.name == "ollama":
>         from .backends.ollama import _preflight_ollama
>         _preflight_ollama(self._model)
>         return
>     self._backend.preflight()
> ```

Also update `AgenticRun._compose_implementer_prompt` to use the Ollama prompt
when the backend is ollama:

```python
def _compose_implementer_prompt(self) -> str:
    from .agent_prompts import IMPLEMENTER_SYSTEM_PROMPT_OLLAMA
    workspace = (self._study_dir / "workspace").resolve()
    preamble = WORKSPACE_PREAMBLE_TEMPLATE.format(workspace_dir=workspace)
    base = (
        IMPLEMENTER_SYSTEM_PROMPT_OLLAMA
        if self._backend.name == "ollama"
        else IMPLEMENTER_SYSTEM_PROMPT
    )
    return preamble + base
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
uv run pytest tests/agentic/test_ollama_backend.py -v
```

Expected: all tests PASS (no real Ollama server needed — all mocked).

- [ ] **Step 5: Run the full test suite**

```bash
uv run pytest tests/agentic/ -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/f3dasm/_src/agentic/backends/ollama.py \
        src/f3dasm/_src/agentic/agent_runtime.py \
        tests/agentic/test_ollama_backend.py
git commit -m "feat(agentic): add Ollama backend with bash-tool Implementer"
```

---

### Task 7: Export `OLLAMA_BACKEND` and add study `config.yaml` files

**Files:**
- Modify: `src/f3dasm/agentic/__init__.py`
- Create: `studies/agentic_black_box_8d/config.yaml`
- Create: `studies/agentic_fragile_becomes_supercompressible/config.yaml`
- Create: `studies/agentic_modular_resonance/config.yaml`
- Create: `studies/agentic_project_euler_078/config.yaml`

- [ ] **Step 1: Export `OLLAMA_BACKEND`**

In `src/f3dasm/agentic/__init__.py`, add:

```python
from .._src.agentic.backends.ollama import OLLAMA_BACKEND
```

Add `"OLLAMA_BACKEND"` to `__all__`.

- [ ] **Step 2: Add `config.yaml` to each agentic study**

`studies/agentic_black_box_8d/config.yaml`:
```yaml
model: claude-haiku-4-5-20251001
backend: claude
budget: "02:00:00"
checkpoint_every: 30
```

`studies/agentic_fragile_becomes_supercompressible/config.yaml`:
```yaml
model: claude-haiku-4-5-20251001
backend: claude
budget: "04:00:00"
checkpoint_every: 30
```

`studies/agentic_modular_resonance/config.yaml`:
```yaml
model: claude-haiku-4-5-20251001
backend: claude
budget: "01:00:00"
checkpoint_every: 30
```

`studies/agentic_project_euler_078/config.yaml`:
```yaml
model: claude-haiku-4-5-20251001
backend: claude
budget: "00:30:00"
checkpoint_every: 30
```

- [ ] **Step 3: Verify import works**

```bash
uv run python -c "from f3dasm.agentic import OLLAMA_BACKEND, StudyConfig; print('ok')"
```

Expected: `ok`.

- [ ] **Step 4: Run full test suite**

```bash
uv run pytest tests/agentic/ -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/f3dasm/agentic/__init__.py \
        studies/agentic_black_box_8d/config.yaml \
        studies/agentic_fragile_becomes_supercompressible/config.yaml \
        studies/agentic_modular_resonance/config.yaml \
        studies/agentic_project_euler_078/config.yaml
git commit -m "feat(agentic): export OLLAMA_BACKEND; add config.yaml to all agentic studies"
```
