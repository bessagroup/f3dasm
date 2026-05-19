# Agentic f3dasm — Architecture

**Updated:** 2026-05-19
**Evidence base:** `literature-map.md` (17 papers, all read in full)

ULTIMATE GOAL: Fully automate the whole interaction a human has with f3dasm. Pain point: non-coders find it hard to learn and use the whole machinery created with f3dasm. Solution: embed a multi-agent system on top of f3dasm principles to solve the pain point. All four pillars of f3dasm (Domain definition, Data Generation, Machine Learning, Optimization) should be swappable and actionable from agentic-f3dasm.

This document defines the contracts every implementation must satisfy. Problem-specific content lives in `supercompressible-baseline.md`.

---

## Vision

We are building agentic-f3dasm, an extension of f3dasm where N persistent agent sessions drive the entire design-of-experiments cycle on behalf of a non-coder. The user's only required input is a `PROBLEM_STATEMENT.md` file. At Level 1, a **Strategizer** and an **Implementer** do the work: the Strategizer reads the problem, asks 1–3 clarifying questions, forms hypotheses, and issues `Delegate(...)` calls; the Implementer enriches and returns the same typed exchange object. At Level 2, additional agent roles (Proposer, Critic, Explorer, Planner, Workers) are added as peers — each channel uses one symmetric typed class that both agents read from and write to. In all cases, the orchestrator is plain Python so a meta-agent can read and mutate the topology without learning any framework idioms (ADAS-searchability).

The architecture rests on five non-negotiable commitments enforced throughout the rest of this document:

- **f3dasm is first-class and pristine.** `ExperimentData`, `Domain`, `DataGenerator`, `Optimizer`, `Block`, `Sampler` are the ground truth. The agentic layer **does not modify any native f3dasm declarations** — it ships as a strictly additive subpackage (`src/f3dasm/agentic/` and supporting new modules under `src/f3dasm/_src/`). No edits to `core.py`, `experimentdata.py`, `domain.py`, `experimentsample.py`, `samplers.py`, `optimizer_factory.py`, the existing optimizer implementations, or the top-level `f3dasm` package `__init__.py`. Users opt in via `from f3dasm.agentic import …`.
- **N-agent peer topology.** Agents are long-running sessions instantiated once at bootstrap; they are peers — no parent/child, no subagent spawning. The runtime routes messages between them in plain Python. Level 1 has two peers (Strategizer + Implementer); Level 2 may have N.
- **The orchestrator is the system's scientific policy — and is itself a subject of optimization.** The routing rules, topology, checkpoint cadence, and delegation contracts encoded in `AgenticRun` represent a hypothesis about how to do computational science efficiently. That hypothesis can be tested: ADAS-style meta-search can mutate and evaluate orchestrators using the same `execute()` entry point; Reflexion-style improvement can refine routing decisions from run histories; multi-armed bandit selection can choose between topologies based on outcome. The ultimate target is a system that uses itself to discover better versions of itself — agentic-f3dasm running agentic-f3dasm, with `ExperimentData` as the shared epistemic record across both object-level and meta-level runs. This is enforced by architectural choice: no framework dependencies, no opaque graph engines, orchestrator stays plain Python so it can be read, mutated, and evaluated without ceremony.
- **Provenance and interpretability are deliverables.** Every Implementer delegation is bracketed by a git commit against the study directory. The commit message is derived from the prior `Report`'s `conclusions`. A non-agentic human can reconstruct the full run history from `git log`. Silent failure is forbidden — the Implementer's `Report` must carry every anomaly.
- **Problem-agnostic core.** Nothing in `src/f3dasm/` (including the new agentic modules) carries problem-specific names, enums, or hardcoded values. All domain-specific content lives in `studies/<study>/PROBLEM_STATEMENT.md` and any resource files it references.

---

## Architecture

### Level 1 — The Two Agents

**Strategizer.** A long-running Claude Agent SDK session that persists for the entire run. It reads files, writes `.md` notes to `runs/<timestamp>/strategizer_notes/`, asks the user clarifying questions via `Ask(question)`, issues `Delegate(intent, expected_report)` calls that block until the Implementer returns a `Report`, and terminates with `Done(summary)`. The Strategizer cannot write code, cannot run computation, and cannot touch `workspace/` directly. Its first action on every run is the briefing-clarification ritual: read `PROBLEM_STATEMENT.md` and all referenced files, then call `Ask` with 1–3 pressing questions before forming any hypothesis.

**Implementer.** A Claude Agent SDK session that receives a `Task(intent, expected_report)` per delegation, executes it using the SDK's built-in file/exec tools (`Read`, `Write`, `Edit`, `Bash`, `Glob`, `Grep`) restricted to the study directory, and returns a structured `Report(actions_taken, files_touched, conclusions, numbers)`. The Implementer owns `workspace/` — its scratch and experiment code that persists across delegations and across runs. It does not form hypotheses, propose research directions, or extend the scope of a task. **The Implementer session is reset at every checkpoint**: a fresh session is instantiated with the Strategizer's checkpoint summary as its new briefing context. This bounds context bleed and limits accumulated tool-call history.

**Tool boundary (what each agent cannot do):**

| | Strategizer | Implementer |
|---|---|---|
| Write code / run computation | No | Yes |
| `Write` to `workspace/` | No | Yes |
| `Delegate` to the other agent | Yes (via runtime) | No |
| `Done` / end the run | Yes | No |
| Form hypotheses / change scope | Yes | No |

### Level 2 — N-Agent Topology

Level 2 generalises the two-agent peer pattern to N agents without changing the orchestrator's fundamental character. Each pair of agents that communicates defines its own **typed contract**: a pair of Python dataclasses (one for the call, one for the response) following the `Task`/`Report` pattern. The orchestrator routes calls between agents in plain Python; no framework is required.

A Level 2 topology has three components:

1. **Agent sessions.** Each role is an `ImplementerSession` or `StrategizerSession` (or a new role-specific Protocol, also with `send(str) -> str`). Sessions are instantiated at bootstrap, run for their designated scope, and reset at checkpoints.
2. **Typed channels.** Every agent-to-agent communication uses a pair of typed dataclasses. The fields are domain-appropriate: structured data over generic content blobs. See *Typed Contracts* below.
3. **Firing primitives.** The orchestrator calls agents using one of three composable utilities: `parallel` (fan-out), `retry` (persistence loop), or `rounds` (fixed-N debate). See *Firing Primitives* below.

**ADAS-searchability.** Because the orchestrator is a plain Python function over `ExperimentData`, a meta-agent can mutate the topology by calling `exec(new_forward_code)` and replacing the function. No framework-specific DSL is needed. The archive of topologies is a list of `(forward_code, score)` tuples; selection, crossover, and mutation are all ordinary Python operations.

### Typed Contracts

Each agent-to-agent channel is governed by **one symmetric typed class** — not a request type and a separate response type. Both agents read from and write to the same dataclass: the initiating agent fills the request fields; the responding agent fills the response fields on the same object and returns it. The contract is enforced in two places simultaneously: the runtime validates deterministically (required fields, type coercion, markdown parsing), and the system prompt instructs each agent explicitly which fields it must populate (e.g. the Implementer's system prompt specifies that `files_touched` is required and must list every path written).

The Strategizer ↔ Implementer channel uses `Delegation`, the canonical example:

```python
@dataclass
class Delegation:
    # Strategizer fills before sending (required)
    intent: str
    expected_report: str
    remaining_time: timedelta | None = None   # wall-clock budget remaining
    budget: timedelta | None = None           # total budget (for 20% warning)

    # Implementer fills before returning (required; validated by parser)
    actions_taken: str = ""
    files_touched: list[str] = field(default_factory=list)
    conclusions: str = ""                     # → git commit message
    numbers: dict[str, Any] = field(default_factory=dict)  # results by name
    raw: str = ""                             # full ## Report block (provenance)

    # Either party may add channel-specific data without schema changes
    metadata: dict[str, Any] = field(default_factory=dict)
```

`Task` and `Report` are retained in the public API as backward-compatible aliases (`Task = Delegation` prefilled with response fields empty; `Report = Delegation` with request fields carried forward). All new Level 2 channels follow the same pattern: one class, both parties enrich it, no pair of separate types. Fields not relevant to a channel may be absent; `metadata` covers any channel-specific additions without breaking the deterministic validation contract.

**Why not generic envelopes.** ADAS's `Info = namedtuple('Info', ['name', 'author', 'content', 'iteration_idx'])` and PABLO's raw JSON strings are appropriate for general-purpose academic frameworks. f3dasm operates in a specific scientific engineering domain: `numbers` lets the runtime extract results by name for git commit messages; `conclusions` is the commit message source; `remaining_time` and `budget` together trigger a 20% warning injection in `_format_task()` so the Implementer scopes its work without parsing text. `metadata` handles the extensibility these generic frameworks achieve with a blob — but without losing the deterministically-checked named fields.

**Serialisation contract.** The request half of a `Delegation` is serialised as `## Task` / `### Intent` / `### Expected report` markdown before being sent over `send(str) -> str`. When fewer than 20% of the budget remains, a `⚠ Budget nearly exhausted` warning line is injected. The response half is parsed from `## Report` / `### Actions taken` / `### Files touched` / `### Conclusions` / `### Numbers` blocks. New channels define their own analogous heading pairs; the one-class-per-channel principle requires that the request and response headings are sections of the same markdown document, not separate messages.

### Typed Stores

Three concrete typed stores serve as shared state in Level 2 topologies. They are not generic `Protocol(read, write)` interfaces — they are domain-specific data structures whose fields the orchestrator and agents explicitly depend on.

**`ExperimentData` (f3dasm core — epistemic center).** The primary shared state for the BO loop. Holds the complete record of inputs (`Domain`), measured outputs, sampling history, optimizer state, and iteration metadata. Every `DataGenerator`, `Optimizer`, and `Sampler` is built to operate on `ExperimentData`. Level 2 agents (Explorer, Planner, Workers) pass `ExperimentData` through the orchestrator so the BO loop's epistemic center is never split across ad-hoc dicts or files. A generic `Store(read, write)` Protocol would strip all of this structure; Level 2 agents that share BO state share `ExperimentData`.

**`AnalysisBase` (AgenticSciML-inspired).** A hierarchical store of per-solution analysis reports. Each node holds a text summary and optional plot paths for one candidate solution in the solution tree. `AnalysisBase.get(solution_id)` returns the **parent** report (lineage), all **sibling** reports (other children of the same parent, i.e. what else was tried at the same generation), and the **uncle** reports (children of the parent's siblings — what the parent's competitors produced). This three-tier slice gives a Critic or SelectorEnsemble exactly the comparative context it needs without reading the entire tree, bounding context while preserving lineage. A flat list of all analyses forces O(N) reading; a full tree includes irrelevant distant branches.

**`TaskRegistry` (PABLO-inspired).** A capped registry of reusable operator descriptions. Each entry carries `{task_name, task_text, attempts, successes, success_rate}`. The registry is capped at `MAX_TASKS = 20` entries; default operators are never pruned. The Planner queries the registry to select proven operators for new Worker delegations; the Workers update `attempts`, `successes`, and `success_rate` after each task. The `success_rate` field lets the Planner weight operator selection toward historically effective approaches — a richer signal than a plain task list.

### Firing Primitives

Three composable utilities for Level 2 orchestrators. All are ordinary Python functions — no graph engine, no decorator magic.

**`parallel(agents, task_fn) -> list[Report]`.** Fan-out: for each agent `a_k` in `agents`, calls `a_k.send(task_fn(k))` concurrently using `concurrent.futures.ThreadPoolExecutor`. Collects and returns typed `Report` objects in the same order as `agents`. PABLO's `K × M` Worker pattern is a `parallel` call with `K × M` sessions. If any call raises or returns an unparseable reply, the exception is wrapped in a failed `Report` (fail loudly — no silent drop).

**`retry(agent, task, *, is_success, max_fails) -> Report`.** Persistence loop: sends the formatted task to `agent.send()`, tests the parsed `Report` against `is_success`, retries if it fails, and raises `AgenticRunError` after `max_fails` consecutive failures. PABLO's Explorer uses `MAX_FAILS = 3`. `is_success` is a caller-supplied predicate on the `Report` (e.g. `lambda r: float(r.numbers.get("best_score", -inf)) > threshold`), giving the calling agent full control over the acceptance criterion.

**`rounds(agent_a, agent_b, n, initial) -> str`.** Fixed-N debate: alternates `n` turns between `agent_a` and `agent_b`, starting with `initial` passed to `agent_a`. Returns the final response after all `n` complete rounds. AgenticSciML's Proposer ↔ Critic debate uses `n = 4` rounds with distinct phases: rounds 1–2 are analysis-only, round 3 is synthesis, round 4 is final. Phase annotations may be embedded in the `initial` argument for phase-aware agents.

### The Runtime

`AgenticRun` is the orchestrator class. It does exactly five things:

**1. Bootstrap.** Validate that `<study-dir>/PROBLEM_STATEMENT.md` exists (raise `AgenticRunError` if missing). Load `config.yaml` if present (`StudyConfig`). Create `<study-dir>/runs/<timestamp>/strategizer_notes/` and `<study-dir>/workspace/`. Run `git init --bare` and an initial empty commit against the study directory. Instantiate both SDK sessions with the system prompts from `agent_prompts.py`.

**2. Strategizer-first dispatch.** Pass the `PROBLEM_STATEMENT.md` text as the opening user message to the Strategizer. The SDK drives the full multi-turn conversation — tool calls, results, and follow-up responses — in one `send()` call.

**3. Route delegations.** When the Strategizer calls `Delegate(intent, expected_report)`:
   - Check wall-clock budget; return `BUDGET_EXHAUSTED` and set `_done_called` if exhausted.
   - Synthesise a git commit message from the prior `Report`'s `conclusions` (or `"initial commit"` for the first delegation).
   - Run `git add -A && git commit` against `<study-dir>` using the run's isolated `--git-dir`.
   - Build a `Task(intent, expected_report, remaining_time, budget)` and render it via `_format_task()`.
   - Forward the rendered markdown to the Implementer's `send()`.
   - Parse the reply for a `## Report` section. On first failure, send `IMPLEMENTER_REPORT_RETRY_PROMPT` and retry once. On second failure, run `_classify_failed_implementer_response()` (Reflexion-style 4-tier diagnosis) and return the `REFLECT:` string to the Strategizer without incrementing the counter.
   - On success, inject `remaining_time` into the `Report`, update `_last_report`, increment the counter, and record JSONL transcripts.

**4. Checkpoint at counter == `CHECKPOINT_EVERY` (default: 30).** Inject `CHECKPOINT_STRATEGIZER_PROMPT` into the Strategizer; print its structured `## Checkpoint` summary to stdout. Block on stdin for the user's input: empty = continue (with any typed text forwarded as steering); `stop` = end the run. Reset the counter. Instantiate a fresh Implementer session with the checkpoint summary as its new briefing (via `IMPLEMENTER_RESET_PROMPT_TEMPLATE`).

**5. Finalise** on `Done(summary)` or user-typed `stop`. Assemble the deliverable folder at `runs/<timestamp>/deliverable/`.

### End-of-Run Signals

Two paths, both fully supported:

- **`Done(summary)`** — Strategizer calls this tool after identifying a best design and completing at least one falsification attempt. The `summary` becomes the `solution.md` body.
- **User-typed `stop`** at any checkpoint prompt — runtime sets `_done_called = True` and proceeds to deliverable assembly. The last checkpoint summary becomes the solution body.

### Folder Layout

```
studies/<study>/
  PROBLEM_STATEMENT.md          # the only file the user must author
  config.yaml                   # optional: model, backend, budget, checkpoint_every
  data.csv                      # optional, user-provided
  sim.py                        # optional, user-provided
  workspace/                    # Implementer scratch; persists across runs
  runs/<timestamp>/
    .git/                       # this run's git repo (orchestrator-managed)
    strategizer_notes/          # Strategizer's .md lab notebook
    transcripts/                # JSONL event logs per delegation
    run.log                     # structured log (HH:MM:SS timestamps)
    deliverable/                # populated at end of run
      solution.md
      git_log.txt
      replication/              # snapshot of workspace/ contents
      transcripts/              # copy of run transcripts
      run.log
```

---

## Evidence Behind Key Decisions

**Peer agents over hierarchical subagents.** Every multi-agent paper in the literature map uses peers, not runtime-spawned subagents: Virtual Lab's PI + scientist team communicates via meeting transcripts; AgenticSciML's Retriever/Proposer/Critic/Engineer are peers in N-round debate; Multi-Agent BO's Strategy + Generation agents are sequential peers; PABLO's Planner/Workers/Explorer are peers. None of these uses runtime agent spawning. We adopt the same pattern: N instantiated sessions, orchestrator routes between them in plain Python. The Claude SDK's `Agent` tool as a parent-child mechanism is excluded by design.

**Typed contracts over generic envelopes.** ADAS uses `Info = namedtuple('Info', ['name', 'author', 'content', 'iteration_idx'])` — a generic envelope whose `content` field carries everything from code snippets to performance numbers to multi-paragraph rationales. PABLO passes raw JSON strings. Both designs are appropriate for academic frameworks optimising for generality. f3dasm optimises for scientific engineering: `Task.expected_report` lets the Strategizer state exactly what it needs; `Report.numbers` lets the runtime extract results by name for git commit messages; `Report.remaining_time` is injected automatically; `Task.budget` and `Task.remaining_time` together trigger the 20% budget warning in `_format_task()` so the Implementer scopes its work without parsing text. Structured fields are a feature of the domain, not ceremony to be eliminated.

**`ExperimentData` as epistemic center, not generic `Store`.** A generic `Protocol(read, write)` store would let any agent write arbitrary keys into shared state, turning it into an untyped bag. f3dasm's `ExperimentData` carries the complete record of a design-of-experiments run: `Domain`, sampled inputs, measured outputs, optimizer state, and iteration metadata. Every `DataGenerator`, `Optimizer`, and `Sampler` is built to operate on `ExperimentData`. Level 2 agents that share BO state share `ExperimentData` — not because we prohibit other state, but because that is where f3dasm's four pillars operate and where the structure is already right.

**Strategy-as-natural-language-task.** In v2 there is no registry of named strategies. The Strategizer delegates via a free-text `intent` field; the Implementer is a full agent that decides how to implement it. AgenticSciML's Engineer/Critic split demonstrates that separating deliberation from implementation — and giving the implementer full coding capability — produces better outcomes than pre-enumerating a fixed operator vocabulary. PABLO's Planner Agent maintains a Task Registry of reusable operators that Worker Agents instantiate; our `Delegate` pattern captures the same delegation flow without requiring the registry to be pre-specified. The Implementer's f3dasm primer (in `IMPLEMENTER_SYSTEM_PROMPT`) provides the vocabulary the agent needs to invoke samplers, optimizers, and data generators correctly.

**ADAS-searchability via plain Python.** ADAS demonstrates that meta-search over agentic system designs is tractable when the system is pure Python: `exec(forward_str)` replaces the `forward` function and every candidate topology can be evaluated without recompiling or reconfiguring a framework. LangGraph topologies are not ADAS-searchable because they require framework-specific node/edge declarations; crewAI and AutoGen topologies require agent role re-registration. Our orchestrator is a plain Python function `forward(ExperimentData) -> ExperimentData` — meta-agents can read it, mutate it, and evaluate it using the same `AgenticRun.execute()` entry point.

**Per-delegation git commits as provenance.** There is no direct paper precedent for this mechanic. We adopt it because git is the tooling a human researcher already uses for code: a `git log --oneline` of the run directory is an immediately readable account of what the Implementer changed and why. Each commit message is derived from the prior `Report`'s `conclusions` field, so the log reads as a condensed experiment narrative without requiring any additional tooling.

**Briefing-clarification ritual.** Virtual Lab's PI agent generates scientist prompts from its own system prompt and runs structured meeting agendas before any candidate generation — a briefing-first pattern. BORA's hypothesis `Comment` objects accumulate named rationales from the start of the run; SelfAI's Cognitive Agent runs an explicit task-parsing stage (stage 1 of its 3-stage pipeline) before any candidate generation. We adopt the same principle as a non-negotiable first step: `Ask` with 1–3 pressing questions before any `Delegate`. This is not courtesy — it is the mechanism by which ambiguities in `PROBLEM_STATEMENT.md` are resolved before they corrupt later delegations.

**Checkpoint with user steering.** SelfAI's adaptive stopping judgment emits a binary stop flag at every iteration based on four explicit criteria; the framework's evidence shows LLM optimizers benefit from explicit pause points rather than running to a hard budget blindly. We add a non-LLM steering layer at those pauses: the user can inject new direction or abort. The checkpoint is at 30 delegations (not evaluated adaptively) because the MVP's primary failure mode is drift, not premature termination.

**Implementer session reset at checkpoint.** Long-running sessions accumulate tool-call history that can produce context bleed — the Implementer imports prior task framing into unrelated tasks. Reflexion's core mechanic is that the Actor does not see raw prior episode histories across trials, only distilled verbal summaries; the self-reflection is the handoff artifact. We adopt the same pattern at a coarser grain: the Implementer session is discarded at checkpoint, and the Strategizer's checkpoint summary (which cites specific Report numbers) becomes the new Implementer's sole briefing. The `workspace/` directory persists, so prior computation artifacts are still accessible.

**Reflexion-style failure classification.** When an Implementer reply cannot be parsed into a `Report`, the runtime does not silently return `ERROR` — it runs `_classify_failed_implementer_response()`, a 4-tier diagnosis that identifies whether the response was too short, contained a capability-limit phrase, had a `## Report` heading with missing subsections, or had no heading at all. The resulting `REFLECT:` string gives the Strategizer a structured explanation of what went wrong, enabling it to rephrase or escalate rather than blindly retrying the same task.

**`AnalysisBase.get(id)` returns parent + siblings + uncles.** A flat list forces agents to read O(N) reports. A full tree includes irrelevant distant branches. The three-tier slice (parent + siblings + uncles, directly from AgenticSciML's AnalysisBase design) bounds context while preserving lineage: the parent shows how the current candidate was derived; the siblings show what else was tried at the same generation; the uncles show what the parent's competitors produced. This is the minimum context for a well-grounded comparative judgment.

**Fail loudly.** BORA's hardcoded GP+EI fallback when LLM mode fails is the failure mode we avoid. Silent fallback produces plausible-looking results that were not agent-generated and masks whether the system is working. The Implementer must surface every anomaly in `### Conclusions`; the runtime returns an explicit `REFLECT:` diagnosis to the Strategizer when no `## Report` block is found after retry; `Done` requires a prior falsification attempt.

---

## Reference Topologies

These pseudocode descriptions show how the PABLO and AgenticSciML topologies would be expressed using f3dasm's typed contracts and firing primitives. They are design targets, not yet implemented.

### PABLO-style Topology

Derived from PABLO (arXiv:2601.22382). Optimises a global candidate pool via repeated Explorer → Planner → K×M Workers cycles.

```python
# One symmetric class per channel — both parties enrich it
@dataclass
class WorkerDelegation:
    # Planner fills (request)
    task_name: str
    task_text: str
    seed: int
    candidates: list[dict[str, Any]]
    remaining_time: timedelta | None = None
    budget: timedelta | None = None

    # Worker fills (response; validated by parser)
    result_candidates: list[dict[str, Any]] = field(default_factory=list)
    improved: bool = False
    conclusions: str = ""                      # → TaskRegistry success tracking
    numbers: dict[str, float] = field(default_factory=dict)  # best_score, n_improved, …
    metadata: dict[str, Any] = field(default_factory=dict)

# Orchestrator (plain Python — ADAS-searchable)
def forward(data: ExperimentData) -> ExperimentData:
    registry = TaskRegistry()
    while remaining > timedelta(0):
        # Explorer: retry until new candidates found or MAX_FAILS=3
        explorer_report = retry(
            explorer,
            format_explorer_task(data),
            is_success=lambda r: len(r.new_candidates) > 0,
            max_fails=3,
        )
        # Planner: select K tasks from TaskRegistry (weighted by success_rate)
        plan = planner.send(format_plan_request(explorer_report, registry))
        # K×M Workers: parallel fan-out
        worker_results = parallel(
            workers,
            lambda k: format_worker_delegation(plan[k], worker_delegations[k]),
        )
        # Update global pool: top-8 + 12 uniform rank sample = 20 candidates
        data = _update_candidate_pool(data, worker_results, top_n=8, sample_n=12)
        registry.update(worker_results)  # uses .conclusions + .improved
    return data
```

### AgenticSciML-style Topology

Derived from AgenticSciML (arXiv:2511.07262). Evolves a solution tree via Proposer ↔ Critic debate, validated by Engineer ↔ Debugger, ranked by ResultAnalyst and SelectorEnsemble.

```python
# One symmetric class per channel
@dataclass
class EngineerDelegation:
    # Orchestrator fills (request)
    design: ExperimentData
    mode: str                                  # "validate" (1 epoch) | "train" (full)
    remaining_time: timedelta | None = None
    budget: timedelta | None = None

    # Engineer fills (response)
    trained_design: ExperimentData | None = None
    conclusions: str = ""
    numbers: dict[str, float] = field(default_factory=dict)  # final_score, epochs, …
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass
class SolutionNode:
    design: ExperimentData
    score: float
    analysis: str
    parent_id: str | None = None

# Orchestrator
def forward(data: ExperimentData) -> ExperimentData:
    analysis_base = AnalysisBase()
    solution_tree: list[SolutionNode] = []
    for generation in range(MAX_GENERATIONS):
        # Retriever: gather relevant prior work and context
        context = retriever.send(format_retrieval_request(data, analysis_base))
        # Proposer ↔ Critic: 4-round debate
        # Rounds 1–2: analysis-only; round 3: synthesis; round 4: final verdict
        debate = rounds(proposer, critic, n=4,
                        initial=format_proposal(data, context, phase="analysis"))
        # Engineer ↔ Debugger: validate (1 epoch) then train (full)
        validated = retry(
            engineer,
            format_engineer_delegation(EngineerDelegation(debate.proposed_design, mode="validate")),
            is_success=lambda r: "error" not in r.conclusions.lower(),
            max_fails=3,
        )
        trained = engineer.send(
            format_engineer_delegation(
                EngineerDelegation(validated.trained_design, mode="train")
            )
        )
        # ResultAnalyst: score + structured analysis; same symmetric pattern
        node = SolutionNode(
            design=trained.trained_design,
            score=float(trained.numbers["final_score"]),
            analysis=result_analyst.send(
                format_analysis(trained_report, analysis_base)
            ),
            parent_id=best.id if solution_tree else None,
        )
        solution_tree.append(node)
        analysis_base.add(node)
        # SelectorEnsemble: pick best parent for next generation
        # get(id) returns parent + siblings + uncles — bounded comparative context
        best = selector.send(format_selection(solution_tree, analysis_base))
        data = best.design
    return data
```

---

## Public API Surface

```python
from f3dasm.agentic import (
    # Level 1 (implemented)
    AgenticRun,          # runtime entry point
    AgenticRunError,     # raised for non-recoverable orchestrator failures
    CHECKPOINT_EVERY,    # Implementer-call cadence (default: 30)
    LookupDataGenerator, # nearest-neighbour pool evaluator the Implementer may import
    MVP_DEFAULT_MODEL,   # "claude-haiku-4-5-20251001"
    Report,              # dataclass: actions_taken, files_touched, conclusions,
                         #             numbers, raw, remaining_time
    Task,                # dataclass: intent, expected_report, remaining_time, budget
    StudyConfig,         # dataclass: model, backend, budget, checkpoint_every
    Backend,             # frozen dataclass: name, default_model, strategizer_factory,
                         #                   implementer_factory, preflight
    CLAUDE_BACKEND,      # default Claude Agent SDK backend instance
    OLLAMA_BACKEND,      # Ollama backend instance (local models, bash-tool Implementer)

    # Level 2 (typed stores + firing primitives — not yet implemented)
    # Delegation,        # symmetric exchange class for Strategizer↔Implementer channel
    # AnalysisBase,      # hierarchical analysis store; .get(id) → parent+siblings+uncles
    # TaskRegistry,      # operator registry: {task_name, task_text, attempts,
    #                    #                     successes, success_rate}, MAX=20
    # parallel,          # fan-out: dispatch task_fn(k) to K agents, collect Delegations
    # retry,             # persistence loop: retry until is_success(d) or max_fails
    # rounds,            # fixed-N debate: alternate n turns between two agents
)
```

CLI invocation:

```
python -m f3dasm.agentic <study-dir> [--model MODEL] [--checkpoint-every N] [--backend BACKEND]
```

The study directory must contain `PROBLEM_STATEMENT.md`. The command prints the absolute path of the `deliverable/` folder on exit.

---

## Per-Study Layout

The only file the user must author is `PROBLEM_STATEMENT.md`. Everything else is optional or auto-created by the runtime.

| Path | Required | Owner |
|---|---|---|
| `PROBLEM_STATEMENT.md` | Yes | User |
| `config.yaml` | No | User (`model`, `backend`, `budget`, `checkpoint_every`) |
| `data.csv` | No | User (Implementer may read) |
| `sim.py` | No | User (Implementer may import) |
| `workspace/` | Auto-created | Implementer scratch |
| `runs/<timestamp>/` | Auto-created | Runtime |
| `runs/<timestamp>/strategizer_notes/` | Auto-created | Strategizer writes |
| `runs/<timestamp>/transcripts/` | Auto-created | Runtime records JSONL events |
| `runs/<timestamp>/deliverable/` | Auto-created at end | Runtime assembles |

`PROBLEM_STATEMENT.md` should describe the problem in the same terms a domain expert would use on day 1 of an unsolved problem: what the variables are, what the objective is, what data is available, what constraints apply. It should not leak the answer.

---

## Deliverable Folder

`runs/<timestamp>/deliverable/` is the canonical hand-off artifact. Contents:

| File | Contents |
|---|---|
| `solution.md` | Strategizer's final `Done(summary)` text (or last checkpoint summary); run metadata (model, total delegations, total turns, timestamp, budget, time used); last 30 lines of git log as a provenance block. |
| `git_log.txt` | Full `git log --pretty=format:"%h %ai %s"` of the run repo — one line per delegation commit. |
| `replication/` | Snapshot of `workspace/` at end of run — all scripts, CSVs, and artefacts the Implementer produced. |
| `transcripts/` | JSONL event logs: `strategizer.jsonl` (every tool call + result) and per-delegation `{n}_implementer.jsonl` files. |
| `run.log` | Structured run log with HH:MM:SS timestamps — all INFO/WARNING events from the `f3dasm.agentic` logger. |

**Auditability.** A non-agentic human can reconstruct what happened by reading `git_log.txt` (what changed and when), the `replication/` scripts (what code was run), `solution.md` (what the Strategizer concluded), and `transcripts/` (every tool call and result in order). No LLM is required to read the audit trail.

---

## Hard Rules

- No edits to native f3dasm modules. Agentic code lives in `src/f3dasm/agentic/` (public re-exports + CLI) and `src/f3dasm/_src/agentic/` (implementation: `agent_runtime.py`, `agent_prompts.py`, `lookup.py`, and the `backends/` subpackage).
- NumPy-style docstrings everywhere; no inline comments; line length ≤ 79; ruff-clean.
- Default model: `claude-haiku-4-5-20251001` (Claude Agent SDK).
- `Done` requires that a falsification attempt has been delegated and its `Report` reviewed. Calling `Done` without falsification is a Strategizer failure mode; the system prompt enforces this via the `PREMATURE CONVERGENCE` failure mode entry.
- Fail loudly: no silent fallbacks, no swallowed exceptions, no unreported anomalies.
- **One symmetric class per channel.** Every agent-to-agent channel defines a single typed dataclass that both parties enrich. The initiating agent fills request fields; the responding agent fills response fields on the same object. `Message(role, content, metadata)`, `Info(name, author, content, iteration_idx)`, and any untyped generic wrapper are prohibited. Fields must be domain-appropriate and named; `metadata: dict[str, Any]` covers channel-specific extensions without sacrificing deterministic validation of the named fields.
- **`ExperimentData` is the shared BO state, not a generic `Store` Protocol.** Level 2 agents that share design-of-experiments state pass `ExperimentData`. Do not introduce a generic read/write Protocol that strips f3dasm's domain structure.

---

## What Is Out of Scope (Level 3+)

The following are explicitly deferred:

- **`replicate.py` auto-generation** — a script that re-derives the best design without LLMs. The `replication/` folder provides the raw artifacts; the script is not auto-generated.
- **Cost dashboards** — explicit per-run token and dollar accounting.
- **Vector-indexed memory** — embedding-based retrieval over prior runs or prior Reports. The checkpoint summary is the L1/L2 memory mechanism.
- **LLM-graded answer verification** — an LLM judge that checks whether the Strategizer's `Done(summary)` correctly describes the best design found, analogous to a reviewer checking a paper's conclusion against its experimental section.

---

## Critical Files

| File | Role |
|---|---|
| `src/f3dasm/_src/agentic/agent_runtime.py` | `AgenticRun`, `Task`, `Report`, `StudyConfig`, `AgenticRunError`, `CHECKPOINT_EVERY`, `_format_task`, `_parse_report`, `_classify_failed_implementer_response`, git helpers, JSONL transcript recording, deliverable assembly |
| `src/f3dasm/_src/agentic/agent_prompts.py` | All prompt constants: the two system prompts, checkpoint prompt, reset template, path-aware preambles, REFLECT diagnoses, corrective-retry prompt |
| `src/f3dasm/_src/agentic/backends/base.py` | `Backend` frozen dataclass + `StrategizerSession` / `ImplementerSession` Protocols |
| `src/f3dasm/_src/agentic/backends/claude.py` | `_ClaudeStrategizer`, `_ClaudeImplementer`, `_classify_sdk_error`, `_preflight_claude_cli`, `CLAUDE_BACKEND` |
| `src/f3dasm/_src/agentic/backends/ollama.py` | `_OllamaStrategizer`, `_OllamaImplementer`, `_preflight_ollama`, `OLLAMA_BACKEND` |
| `src/f3dasm/_src/agentic/lookup.py` | `LookupDataGenerator` — nearest-neighbour pool evaluator the Implementer may import |
| `src/f3dasm/agentic/__init__.py` | Public re-exports — `from f3dasm.agentic import AgenticRun, Backend, CLAUDE_BACKEND, OLLAMA_BACKEND, …` |
| `src/f3dasm/agentic/__main__.py` | CLI entry — `python -m f3dasm.agentic <study-dir>` |
| `studies/<study>/PROBLEM_STATEMENT.md` | The user's problem statement — the only required per-study file |
| `studies/<study>/config.yaml` | Optional per-study config: `model`, `backend`, `budget`, `checkpoint_every` |
| `studies/<study>/workspace/` | Implementer scratch; persists across delegations and runs |
| `studies/<study>/runs/<timestamp>/strategizer_notes/` | Strategizer's `.md` lab notebook for the run |
| `studies/<study>/runs/<timestamp>/transcripts/` | JSONL event logs: `strategizer.jsonl` + per-delegation `{n}_implementer.jsonl` |
| `studies/<study>/runs/<timestamp>/deliverable/` | End-of-run hand-off: `solution.md`, `git_log.txt`, `replication/`, `transcripts/`, `run.log` |
