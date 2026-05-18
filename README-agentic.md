# agentic-f3dasm

An additive subpackage of [`f3dasm`](./README.md) that drives the full design-of-experiments cycle on behalf of a non-coder, using two persistent Claude Agent SDK sessions and a thin Python runtime. This document is the agentic layer's standalone README; the upstream `f3dasm` package and its top-level `README.md` are untouched.

---

## Summary

`agentic-f3dasm` adds a single CLI entry point on top of f3dasm:

```bash
python -m f3dasm.agentic <study-dir>
```

Inside `<study-dir>` you place one file — `briefing.md` — describing the problem, the design parameters, the objective, and any resources (datasets, simulators) the agent should use. You may also drop optional auxiliary files (CSV pools, Python simulators, reference papers). The runtime then orchestrates two Claude Agent SDK sessions:

- a **Strategizer** that reads everything in the study, asks 1–3 clarifying questions of the user, hypothesises, and delegates concrete tasks;
- an **Implementer** that receives each task, writes and executes Python under a sandboxed `workspace/`, and returns a structured `Report`.

The runtime records every delegation as a git commit against an isolated per-run repository, captures per-turn JSONL transcripts for both agents, and assembles a deliverable folder containing the solution, the supporting code, and the audit trail.

---

## Statement of need

f3dasm is a Python framework that orchestrates the canonical DOE loop: *define a domain → sample candidate designs → evaluate them → optimise → repeat*. The framework itself is powerful but still requires a researcher who can configure a `Domain`, write a `DataGenerator`, pick a sampler, decide when to stop, and interpret the results.

The pain point we address: many users of `f3dasm` (downstream researchers, students, collaborators in non-coding disciplines) carry valuable domain knowledge but are blocked at the framework's surface area. The natural unit of work is the briefing they would already give a graduate-student assistant — a problem statement, constraints, available resources, and a goal — not a Python script.

`agentic-f3dasm` is the smallest possible bridge from a briefing.md to a result. A briefing in, a deliverable out. The agent reads, hypothesises, writes its own code, executes it, falsifies its own claims, and produces a self-contained folder that another human can audit and reproduce without re-running the agent.

The system is built on a literature base of 17 papers covering agentic optimization (BORA, LLAMBO, Multi-Agent BO), multi-agent research systems (Virtual Lab, AgenticSciML, PABLO, Denario), and agentic reflection patterns (ReAct, Reflexion, SelfAI, ADAS). See `docs/specs/literature-map.md` for the full synthesis.

---

## Method overview

The architecture has four pieces, each kept deliberately small.

```text
                ┌──────────────────────────────────┐
   briefing.md ─▶│   AgenticRun                    │── runs/<ts>/strategizer_notes/
                │   (~1 200 lines of Python)       │── runs/<ts>/.git/    (provenance)
                │   - routes messages              │── runs/<ts>/deliverable/
                │   - git commit per delegation    │
                │   - 30-delegation checkpoint     │
                └──┬───────────────────────────┬───┘
                   │                           │
        ┌──────────▼──┐                   ┌────▼─────────────┐
        │ Strategizer │                   │  Implementer     │
        │ (Claude SDK)│                   │  (Claude SDK)    │
        │             │                   │                  │
        │ Read        │  Task(intent,...) │ Read / Write /   │
        │ WriteMd     │  ────────────────▶│ Edit / Bash /    │
        │ Ask         │                   │ Grep / Glob      │
        │ Delegate    │  Report(...)      │ → ## Report      │
        │ Done        │  ◀────────────────│                  │
        └─────────────┘                   └──────────────────┘
              Reads any file                  Owns workspace/
              Writes .md only                 /tmp is forbidden
              Cannot execute code             Cannot redirect strategy
```

**The five non-negotiable commitments** (specified in `docs/specs/architecture.md`):

1. **f3dasm is first-class and pristine.** No native f3dasm modules are edited. The agentic layer ships as `from f3dasm.agentic import …`.
2. **Composable peer topology.** The two agents are peers, not parent/child. Neither spawns subagents at runtime.
3. **The orchestrator is plumbing.** It routes messages, runs git, enforces the checkpoint cadence — nothing more. Strategic decisions belong to the Strategizer.
4. **Provenance and interpretability are deliverables.** Every Implementer delegation is bracketed by a git commit. The deliverable folder includes the solution, the full git log, JSONL transcripts of both agents, and the Implementer's workspace.
5. **Problem-agnostic core.** Nothing in `src/f3dasm/` (including the new agentic modules) carries problem-specific names, enums, or constants. Domain content lives entirely in `studies/<study>/briefing.md`.

**Built-in safeguards engineered into the system prompts:** the Strategizer's prompt names and mitigates anchoring bias, confirmation bias, availability bias, role drift, sycophancy, and premature convergence; the Implementer's prompt enforces a SelfAI-style three-stage reasoning protocol (Restate / Inventory / Plan) before any code execution, and a BORA-style structured `## Report` block on output. A one-shot corrective retry fires when the report can't be parsed; a Reflexion-style `REFLECT:` diagnostic surfaces if the retry also fails.

---

## Code purpose, audience, and goals

| | |
|---|---|
| **Purpose** | Reduce the human labour required to drive an f3dasm DOE study from "write all the Python yourself" to "describe the problem in a markdown file." |
| **Audience** | (a) Domain researchers who use f3dasm but would rather hand off the optimisation loop. (b) Agentic-AI researchers studying how LLM agents do scientific work, with access to a clean codebase and full per-turn transcripts. |
| **Goals (Level 1, this branch)** | A working briefing-driven runtime with verifiable provenance, that handles real numerical optimisation against a representative test problem. |
| **Non-goals (this branch)** | Cost dashboards, multi-Implementer topologies, vector-indexed memory, ADAS-style offline meta-search. These are explicit Level-2+ work. |

---

## Getting started

### Install (with `uv`)

We recommend [`uv`](https://docs.astral.sh/uv/) for environment management; plain `pip install` can silently fall back to the global interpreter if no venv is active, which is a bad fit for a project that ships a runtime depending on a pinned `claude-agent-sdk` version.

```bash
# Inside the f3dasm repo
uv venv                              # creates ./.venv with a compatible Python (>=3.10)
uv pip install -e ".[agentic]"       # editable install + agentic extras into ./.venv
```

The `agentic` extras install `claude-agent-sdk`. You also need the Claude CLI binary on your `PATH`. Through 2026-06-14, Agent SDK calls count against your Claude.ai subscription pool; after that, programmatic usage moves to a separate credit bucket (Pro: $20/mo; Max 20×: $200/mo).

> If you absolutely cannot use `uv`, the legacy alternative is `python -m venv .venv && source .venv/bin/activate && pip install -e ".[agentic]"`. *Do not* run `pip install -e ".[agentic]"` without first activating a venv — that will install into your system Python.

### Run an existing study

```bash
uv run python -m f3dasm.agentic studies/modular_resonance
```

`uv run` executes inside `./.venv` automatically, so you do not have to remember to `source .venv/bin/activate`. The Strategizer will print 1–3 clarifying questions on stdout and wait for your typed answers. After the briefing phase, it delegates work to the Implementer. When the run completes, the deliverable lives at `studies/modular_resonance/runs/<timestamp>/deliverable/`.

### Make your own study

```bash
mkdir studies/my_problem
cat > studies/my_problem/briefing.md <<EOF
# My problem

I want to find (x, y) in [0, 1]² that maximises f(x, y) where
f is implemented in sim.py. Do not run more than 200 evaluations.
EOF
cp my_simulator.py studies/my_problem/sim.py
uv run python -m f3dasm.agentic studies/my_problem
```

That is the entire user-facing surface.

### Studies in this repo

- **`studies/modular_resonance/`** — a two-parameter integer optimisation inspired by Project Euler #952 (multiplicative orders modulo factorials). The agent must build an f3dasm `Domain`, wrap the resonance computation in a `DataGenerator`, sample candidates, and report the global optimum. Independent brute-force confirms the agent's answer for this benchmark: `(k=6, m=99991, resonance ≈ 8685.089)`.
- **`studies/project_euler_078/`** — a simpler smoke test (coin partitions). Demonstrates the agentic loop on a classic problem; the f3dasm DOE machinery does *not* engage here because the problem is a pure recurrence with no parameter space — kept as a baseline reference.
- **`studies/fragile_becomes_supercompressible/`** — the canonical f3dasm benchmark (Bessa 2019 supercompressible metamaterial). The briefing is not yet authored for this study; it remains as the next target.

### CLI flags

```text
python -m f3dasm.agentic <study-dir> \
    [--model claude-haiku-4-5-20251001] \
    [--checkpoint-every 30]
```

---

## Repository layout (agentic-only)

```text
src/f3dasm/agentic/
    __init__.py            # public API: AgenticRun, Task, Report,
                           # AgenticRunError, MVP_DEFAULT_MODEL, …
    __main__.py            # CLI entry point

src/f3dasm/_src/optimization/
    agent_runtime.py       # AgenticRun, _ClaudeStrategizer,
                           # _ClaudeImplementer, tool closures,
                           # git helpers, checkpoint logic
    agent_prompts.py       # the four prompt constants

src/f3dasm/_src/datageneration/
    lookup.py              # LookupDataGenerator — utility the
                           # Implementer may import for
                           # pool-backed evaluation

tests/optimization/
    test_agent_runtime.py  # 30+ tests over the runtime
    test_agent_prompts.py  # prompt-content sanity checks

studies/<study>/
    briefing.md            # the only required file
    (data, sims, papers — optional)
    workspace/             # Implementer scratch (study-level, persists)
    runs/<ts>/             # one folder per agent run
        .git/              # provenance (isolated per-run repo)
        strategizer_notes/ # Strategizer .md lab notebook (run-level)
        transcripts/       # JSONL per-turn transcripts (both agents)
        deliverable/       # what an outsider should read
            solution.md
            replication/   # mirror of workspace/
            git_log.txt
            transcripts/

docs/specs/                # design docs (local-only by convention)
    architecture.md        # the v2 architecture, full spec
    literature-map.md      # 17-paper synthesis
    supercompressible-baseline.md
    checklist.md           # BRG Python Coding Style of Conduct
```

---

## Authorship

- **Elvis Aguero** (`elvis_alexander_aguero_vera@brown.edu`) — design, architecture, implementation.
- Bessa Research Group, Brown University — context and host project.

The agentic-f3dasm layer was developed in collaboration with Claude (Anthropic) — both the design conversations and the implementation passes were carried out with Claude Code as a working partner.

---

## Community support

- **Architecture questions:** open an issue on the [`f3dasm`](https://github.com/bessagroup/f3dasm) repository with the prefix `[agentic]` in the title.
- **Bug reports for the agentic layer:** same channel.
- **Design proposals (Level 2+):** include a citation from the `docs/specs/literature-map.md` paper set where possible.
- **Discussion:** the Bessa Research Group's internal channels.

---

## License

`agentic-f3dasm` inherits the host project's **BSD-3-Clause** license. See `LICENSE` at the repository root.

---

## Status

This is **Level 1 of agentic-f3dasm** — a working, verifiable MVP. It has been exercised end-to-end against the `modular_resonance` study with Haiku 4.5; the deliverable folder is reproducible via the agent-written `replicate.py`, and an independent brute-force solver agrees with the agent's answer on that benchmark.

Level 2 work, in priority order, includes: (1) a hypothesis-log enforcement gate before `Done()` may fire; (2) richer Implementer-internal tool-call capture in transcripts; (3) cost monitoring; (4) multi-Implementer / parallel-meeting topologies (Virtual Lab pattern). See `docs/specs/architecture.md` § *What is not in this architecture* for the full Level-2+ list.
