# Design: Study Config File, Time Budget, and Ollama Backend

**Date:** 2026-05-19
**Features:** (a) time budget enforcement, (d) Ollama/local model backend
**Status:** Approved

---

## Problem

The current `AgenticRun` CLI requires flag arguments (`--model`, `--checkpoint-every`) for
configuration, and has no time-budget support. Non-coder researchers should not need to
touch the CLI at all. Adding `--budget` and `--backend` as additional CLI flags would
worsen an already fragmented UX. The gold standard for project-level tooling is a single
config file per project; CLI args exist only as one-off overrides.

---

## Decision: Single `config.yaml` Per Study

Every study directory may contain a `config.yaml`. The CLI becomes:

```
python -m f3dasm.agentic <study-dir>
```

No required arguments ever. The config file is the canonical source of all runtime
parameters. CLI flags override the file for one-offs. Precedence:

```
built-in defaults  <  config.yaml  <  CLI flags
```

`PROBLEM_STATEMENT.md` is kept purely as the scientific problem description — no YAML
front matter, no runtime metadata.

### config.yaml schema

```yaml
model: claude-haiku-4-5-20251001   # LLM model identifier
backend: claude                    # "claude" | "ollama"
budget: "01:30:00"                 # hh:mm:ss wall-clock budget; omit for unbounded
checkpoint_every: 30               # Implementer-call cadence (default: 30)
```

All keys are optional. An absent `config.yaml` is valid (all defaults apply).

### Config loading

A new `StudyConfig` dataclass is populated by `_load_study_config(study_dir)`:

1. Read `<study-dir>/config.yaml` if it exists; parse with `yaml.safe_load`.
2. Validate known keys; raise `AgenticRunError` on unrecognised keys or malformed values.
3. Apply CLI overrides on top.
4. Return `StudyConfig`.

`AgenticRun.__init__` accepts a `StudyConfig` (or `None` for pure defaults). The CLI
constructs `StudyConfig` before instantiating `AgenticRun`.

---

## Feature (a): Time Budget

### User interface

```yaml
# studies/my_study/config.yaml
budget: "01:30:00"
```

Absent `budget` key → unbounded run (current behaviour, no regression).

### Runtime behaviour

`AgenticRun` gains:

- `_budget: timedelta | None` — parsed from `StudyConfig.budget`.
- `_start_time: datetime` — recorded at bootstrap.
- `_remaining() -> timedelta | None` — `budget - (now - start_time)`, or `None`.

**Before each delegation:** if `_remaining()` is not `None` and `≤ 0`, skip the
delegation and trigger clean shutdown (same code path as user-typed `stop`). The
in-flight delegation always completes — the check happens at the *start* of the next
delegation, not mid-flight.

### `remaining_time` on message dataclasses

`Task` and `Report` each gain a `remaining_time: timedelta | None = None` field.

- The runtime fills `Task.remaining_time` from `_remaining()` when constructing each
  `Task` before sending to the Implementer.
- The runtime fills `Report.remaining_time` from `_remaining()` when forwarding each
  `Report` back to the Strategizer.

Both are rendered in the formatted text blocks the agents read. Format:

```
**Time remaining: 01:12:47**
```

When `remaining_time < 20 % of budget`, an additional warning line is appended to the
Task block:

```
⚠ Budget nearly exhausted — scope remaining work accordingly.
```

Neither agent manages a clock. They read a field injected by the runtime.

### Deliverable metadata

`solution.md` gains a `budget` and `time_used` line in its run-metadata section.

---

## Feature (d): Ollama Backend

### New file: `src/f3dasm/_src/agentic/backends/ollama.py`

Exports `OLLAMA_BACKEND: Backend` — a drop-in replacement for `CLAUDE_BACKEND`.

### Tool surface

**Strategizer tools** (semantic, no side effects — identical intent to Claude version):

| Tool | Args | What the runtime does |
|---|---|---|
| `Ask` | `question: str` | Prints to stdout, blocks on stdin |
| `Delegate` | `intent: str`, `expected_report: str` | Routes to Implementer |
| `Done` | `summary: str` | Triggers deliverable assembly |

**Implementer tool** (single, powerful):

| Tool | Args | What the runtime does |
|---|---|---|
| `bash` | `cmd: str` | `subprocess.run(cmd, shell=True, cwd=study_dir)` |

The Implementer's tool surface collapses to a single `bash` tool. This is sufficient
for all file reads, writes, Python execution, and data processing. The Implementer's
system prompt (Ollama variant) explicitly instructs it to use `bash` for all filesystem
operations.

### Multi-turn loop

Both sessions implement the same manual loop (Claude SDK handles this for the Claude
backend; we own it for Ollama):

```python
while True:
    response = ollama.chat(model=model, messages=history, tools=tool_schemas)
    if response.message.tool_calls:
        for call in response.message.tool_calls:
            result = _execute_tool(call)
            history.append(tool_result_message(call, result))
    else:
        return response.message.content   # final text answer
```

For the Strategizer, `_execute_tool` returns the tool call to the runtime (does not
execute locally). For the Implementer, `_execute_tool` runs `bash` via subprocess.

The loop terminates when the model emits a plain-text response with no tool calls. For
the Implementer this must contain a `## Report` block; the retry/reflect logic from the
Claude backend applies unchanged.

### Preflight

`_preflight_ollama(model)` checks:

1. `ollama` Python package is importable (lazy import, raises `AgenticRunError` if not).
2. Ollama server is reachable (`ollama.list()` succeeds).
3. The requested model is available locally (`model` appears in `ollama.list()`).

### Dependency

`ollama` Python package — optional, imported lazily at first `send()` call. Not added
to `pyproject.toml` hard dependencies; the preflight error message includes the install
command.

### Public export

```python
from f3dasm.agentic import OLLAMA_BACKEND
```

### Default model

`qwen2.5:0.5b` — smallest model with tool-calling support; runs on any laptop. Users
should switch to a larger model (`llama3.1:8b`, `qwen2.5:7b`) for real runs.

---

## CLI changes

```
python -m f3dasm.agentic <study-dir>
                         [--model MODEL]
                         [--backend {claude,ollama}]
                         [--budget HH:MM:SS]
                         [--checkpoint-every N]
```

All flags are optional overrides. The `<study-dir>/config.yaml` is always read first.

---

## Files changed

| File | Change |
|---|---|
| `src/f3dasm/_src/agentic/agent_runtime.py` | `StudyConfig` dataclass; `_load_study_config`; budget fields + `_remaining()`; `Task.remaining_time`; `Report.remaining_time`; pre-delegation budget check; Task/Report formatting |
| `src/f3dasm/_src/agentic/backends/ollama.py` | New file — `_OllamaStrategizer`, `_OllamaImplementer`, `_preflight_ollama`, `OLLAMA_BACKEND` |
| `src/f3dasm/_src/agentic/agent_prompts.py` | Ollama-specific Implementer system prompt (bash-centric tooling instructions) |
| `src/f3dasm/agentic/__init__.py` | Export `StudyConfig`, `OLLAMA_BACKEND` |
| `src/f3dasm/agentic/__main__.py` | Read `config.yaml`; build `StudyConfig`; add `--backend`, `--budget` CLI flags |
| `studies/*/config.yaml` | Add `config.yaml` to each agentic study (with sensible per-study defaults) |

---

## Out of scope

- Multi-backend runs (Strategizer on Claude, Implementer on Ollama) — Level 2.
- Config validation beyond known keys — warn and continue.
- Streaming output from Ollama — blocking `chat()` call only.
