"""Agentic-f3dasm v2 runtime — Strategizer/Implementer orchestrator.

The module owns the thin Python layer that drives two persistent Claude
Agent SDK sessions (a **Strategizer** and an **Implementer**), routes
tool calls between them, records per-delegation git commits, and
assembles the deliverable folder at the end of a run.

The public surface is minimal:

- ``AgenticRun`` — the orchestrator class.
- ``Task`` — dataclass sent to the Implementer on each delegation.
- ``Report`` — dataclass produced by parsing the Implementer's response.
- ``AgenticRunError`` — raised for non-recoverable orchestrator failures.
- ``CHECKPOINT_EVERY`` — Implementer-call cadence (module constant).

Protocol stubs ``StrategizerSession`` and ``ImplementerSession`` let
test code inject deterministic stubs via the factory parameters.
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
import inspect
import json
import logging
import re
import shutil
import subprocess
import sys
import textwrap
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, TextIO

# Local
from .agent_prompts import (
    CHECKPOINT_STRATEGIZER_PROMPT,
    IMPLEMENTER_REPORT_RETRY_PROMPT,
    IMPLEMENTER_RESET_PROMPT_TEMPLATE,
    IMPLEMENTER_SYSTEM_PROMPT,
    REFLECT_DIAGNOSIS_CAPABILITY_LIMIT,
    REFLECT_DIAGNOSIS_DEFAULT,
    REFLECT_DIAGNOSIS_MISSING_SUBSECTIONS_TEMPLATE,
    REFLECT_DIAGNOSIS_NO_REPORT_HEADING,
    REFLECT_DIAGNOSIS_SHORT,
    RUN_PATHS_PREAMBLE_TEMPLATE,
    STRATEGIZER_SYSTEM_PROMPT,
    WORKSPACE_PREAMBLE_TEMPLATE,
)

# Session protocols live in backends.base and are re-exported here so
# existing test code that does
#   ``from f3dasm._src.agentic.agent_runtime import StrategizerSession``
# continues to work without modification.
from .backends.base import (
    AgentRole,
    AgentSession,
    Backend,
    Edge,
    Graph,
    ImplementerSession,
    StrategizerSession,
)

# MVP_DEFAULT_MODEL: keep the canonical value here as well so that
# ``from f3dasm._src.agentic.agent_runtime import MVP_DEFAULT_MODEL``
# continues to work.  The identical string lives in backends.claude too;
# neither module imports it from the other (avoids a circular import).
MVP_DEFAULT_MODEL: str = "claude-haiku-4-5-20251001"
"""Default Claude model used when no model is specified."""

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Elvis Aguero (elvis_alexander_aguero_vera@brown.edu)"
__credits__ = ["Elvis Aguero"]
__status__ = "Experimental"
# =============================================================================

__all__ = [
    "AgenticRun",
    "AgenticRunError",
    "Delegation",
    "Report",
    "RunContext",
    "Task",
    "_format_delegation",
    "_parse_delegation",
    "read_transcript",
    "register_backend",
]

# ---------------------------------------------------------------------------
# Module constant
# ---------------------------------------------------------------------------

CHECKPOINT_EVERY: int = 30
"""Number of Implementer delegations between checkpoints."""

_BACKEND_REGISTRY: dict[str, "Backend"] = {}
"""Registry of named backends for use in StudyConfig.backend."""


def register_backend(name: str, backend: "Backend") -> None:
    """Register a named backend so it can be referenced by StudyConfig.backend.

    Parameters
    ----------
    name : str
        The name used in ``config.yaml`` under the ``backend`` key.
    backend : Backend
        The backend instance to register.
    """
    _BACKEND_REGISTRY[name] = backend

# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Task:
    """Task sent from the Strategizer to the Implementer.

    Parameters
    ----------
    intent : str
        Natural-language description of what to do.
    expected_report : str
        Description of the measurements or conclusions the Report must
        contain.
    remaining_time : timedelta or None
        Wall-clock time remaining in the run budget, if a budget is set.
    budget : timedelta or None
        Total run budget (used to compute the 20% warning threshold).
    """

    intent: str
    expected_report: str
    remaining_time: timedelta | None = None
    budget: timedelta | None = None


@dataclass
class Report:
    """Parsed output from the Implementer's Report block.

    Parameters
    ----------
    actions_taken : str
        Bullet list extracted from ``### Actions taken``.
    files_touched : list[str]
        Paths extracted from ``### Files touched``.
    conclusions : str
        Free-form prose from ``### Conclusions``.
    numbers : dict[str, Any]
        Key-value pairs from ``### Numbers``.
    raw : str
        The full ``## Report`` markdown block as written by the
        Implementer.
    remaining_time : timedelta or None
        Wall-clock time remaining when the report was received.
    """

    actions_taken: str
    files_touched: list[str]
    conclusions: str
    numbers: dict[str, Any]
    raw: str
    remaining_time: timedelta | None = None


@dataclass
class Delegation:
    """Round-trip exchange envelope for an agent-to-agent channel.

    The initiating agent constructs the envelope with a :class:`Task`
    (request half).  After the responding agent replies,
    :func:`_parse_delegation` sets ``report`` to a :class:`Report`
    (response half).  ``is_complete`` is ``True`` once the report is set.

    Parameters
    ----------
    task : Task
        The request half (filled by the initiating agent).
    report : Report or None
        The response half (filled by :func:`_parse_delegation`, ``None``
        until the round-trip completes).
    metadata : dict[str, Any]
        Channel-specific extensions; either party may add keys.
    """

    task: Task
    report: Report | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        """True once the responding agent has filled the report."""
        return self.report is not None


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class AgenticRunError(RuntimeError):
    """Raised for non-recoverable orchestrator failures."""


# ---------------------------------------------------------------------------
# RunContext protocol
# ---------------------------------------------------------------------------

from typing import Protocol as _Protocol  # noqa: E402


class RunContext(_Protocol):
    """Stable interface exposed to custom topology callables.

    A custom topology is a ``Callable[[RunContext], None]`` passed to
    :class:`AgenticRun`.  It calls methods on this context instead of
    depending on :class:`AgenticRun` internals.
    """

    def delegate(
        self, task: Task, target: str = "implementer"
    ) -> Delegation:
        """Send *task* to *target*; return the completed Delegation."""
        ...

    def checkpoint(self) -> None:
        """Commit current run state to git and reset the delegation counter."""
        ...

    def budget_remaining(self) -> timedelta | None:
        """Wall-clock time remaining, or ``None`` if no budget is set."""
        ...

    def done(self, summary: str) -> None:
        """Signal the run is complete with a final summary string."""
        ...

    def ask(self, question: str) -> str:
        """Print *question* to stdout and return the operator's reply."""
        ...

    def parallel(
        self,
        task_fn: "Callable[[int], Task]",
        targets: "list[str]",
    ) -> "list[Delegation]":
        """Call ``task_fn(i)`` and delegate to ``targets[i]`` for each i.

        Delegates are issued sequentially through the run's full pipeline
        (budget check, git commit, transcript recording).  Order is
        preserved.  Errors are captured as Delegations with
        ``metadata["error"] = True``; nothing is silently dropped.

        Parameters
        ----------
        task_fn : callable
            Called with zero-based index; returns the Task for that slot.
        targets : list[str]
            Agent names, one per slot.  Must be declared in the Graph/roles.
        """
        ...

    def retry(
        self,
        task: "Task",
        target: str = "implementer",
        *,
        is_success: "Callable[[Delegation], bool]",
        max_fails: int = 3,
        on_failure: "Callable[[Delegation, int], str] | None" = None,
    ) -> "Delegation":
        """Persistence loop: delegate *task* to *target* until ``is_success``.

        On each failure, sends ``on_failure(delegation, attempt)`` to the
        agent as a corrective message (1-based attempt count).  When
        *on_failure* is ``None``, a brief built-in corrective is used.

        Raises
        ------
        AgenticRunError
            After ``max_fails`` consecutive failures.
        """
        ...

    def debate(
        self,
        target_a: str,
        target_b: str,
        n: int,
        initial: str,
    ) -> "list[Delegation]":
        """Fixed-N debate: alternate *n* rounds between *target_a* and *target_b*.

        Each round's response is used as the next round's input.  Returns
        ``2*n`` Delegations ``[a1, b1, a2, b2, …]``.  All delegations go
        through the full run pipeline (budget, git, transcripts).

        The final response is ``debate(…)[-1].report.conclusions``.
        """
        ...


# ---------------------------------------------------------------------------
# StudyConfig
# ---------------------------------------------------------------------------

_KNOWN_CONFIG_KEYS: frozenset[str] = frozenset(
    {"model", "backend", "budget", "checkpoint_every"}
)


def _parse_budget(value: str) -> timedelta:
    parts = value.strip().split(":")
    if len(parts) != 3:
        raise ValueError(f"expected HH:MM:SS, got {value!r}")
    h, m, s = (int(p) for p in parts)
    return timedelta(hours=h, minutes=m, seconds=s)


@dataclass
class StudyConfig:
    model: str | None = None
    backend: str = "claude"
    budget: timedelta | None = None
    checkpoint_every: int | None = None


def _load_study_config(study_dir: Path) -> StudyConfig:
    import yaml  # soft import

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
    budget = None
    if "budget" in raw:
        try:
            budget = _parse_budget(str(raw["budget"]))
        except (ValueError, TypeError) as exc:
            raise AgenticRunError(
                f"config.yaml: invalid budget {raw['budget']!r} — expected HH:MM:SS ({exc})."
            ) from exc
    return StudyConfig(
        model=raw.get("model"),
        backend=raw.get("backend", "claude"),
        budget=budget,
        checkpoint_every=raw.get("checkpoint_every"),
    )


# ---------------------------------------------------------------------------
# Note: _ClaudeStrategizer, _ClaudeImplementer, _classify_sdk_error,
# _run_async_safe, _infer_schema_from_callable, _IMPLEMENTER_ALLOWED_TOOLS,
# and _IMPLEMENTER_DENY_LIST have all moved to backends.claude.
# They are no longer re-exported from this module; update any direct
# imports to use ``f3dasm._src.agentic.backends.claude`` instead.
# ---------------------------------------------------------------------------


def _parse_report(text: str) -> Report | None:
    """Extract a ``Report`` from an Implementer reply.

    The parser searches for a ``## Report`` heading and extracts the four
    sub-sections by heading.  Returns ``None`` if the heading is absent.

    Parameters
    ----------
    text : str
        Full assistant message text.

    Returns
    -------
    Report or None
    """
    match = re.search(r"##\s+Report\b", text, re.IGNORECASE)
    if not match:
        return None

    raw_start = match.start()
    raw_block = text[raw_start:]

    def _section(heading: str, src: str) -> str:
        pattern = rf"###\s+{re.escape(heading)}\s*\n(.*?)(?=\n###|\Z)"
        m = re.search(pattern, src, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    actions = _section("Actions taken", raw_block)
    files_raw = _section("Files touched", raw_block)
    conclusions = _section("Conclusions", raw_block)
    numbers_raw = _section("Numbers", raw_block)

    files: list[str] = [
        ln.lstrip("- ").strip()
        for ln in files_raw.splitlines()
        if ln.strip() and ln.strip() != "-"
    ]

    numbers: dict[str, Any] = {}
    for ln in numbers_raw.splitlines():
        if ":" in ln:
            key, _, val = ln.partition(":")
            key = key.strip()
            val = val.strip()
            if key:
                try:
                    numbers[key] = float(val)
                except ValueError:
                    numbers[key] = val

    return Report(
        actions_taken=actions,
        files_touched=files,
        conclusions=conclusions,
        numbers=numbers,
        raw=raw_block,
    )


def _truncate_for_commit(text: str, max_len: int = 70) -> str:
    """Return ``text`` truncated to at most ``max_len`` chars.

    Parameters
    ----------
    text : str
        Source string.
    max_len : int
        Maximum character count.

    Returns
    -------
    str
    """
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _preview(text: str, max_len: int) -> str:
    """Return a single-line preview of ``text``, ≤ ``max_len`` chars.

    Newlines collapse to spaces; the result is suffixed with ``…``
    when truncated. Used for human-readable log messages.
    """
    if text is None:
        return ""
    flat = " ".join(text.split())
    if len(flat) <= max_len:
        return flat
    return flat[: max_len - 1].rstrip() + "…"


def _format_elapsed(seconds: float) -> str:
    """Format a duration in seconds as ``M:SS`` (or ``H:MM:SS``)."""
    seconds = int(seconds)
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _run_git(
    args: list[str],
    *,
    git_dir: Path,
    work_tree: Path,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a git command with an isolated ``--git-dir`` / ``--work-tree``.

    Parameters
    ----------
    args : list[str]
        git sub-command and arguments (e.g. ``["add", "-A"]``).
    git_dir : Path
        Path to the ``<run_dir>/.git`` directory.
    work_tree : Path
        Path to the study root used as the git work-tree.
    check : bool
        If True, raise ``subprocess.CalledProcessError`` on failure.

    Returns
    -------
    subprocess.CompletedProcess
    """
    cmd = [
        "git",
        f"--git-dir={git_dir}",
        f"--work-tree={work_tree}",
    ] + args
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=check,
    )


# ---------------------------------------------------------------------------
# Reflexion-style failure classifier (Piece B)
# ---------------------------------------------------------------------------


_REQUIRED_SUBSECTIONS: list[str] = [
    "Actions taken",
    "Files touched",
    "Conclusions",
    "Numbers",
]

_CAPABILITY_PHRASES: list[str] = [
    "i don't have",
    "i cannot",
    "i am unable",
]


def _classify_failed_implementer_response(
    response_text: str,
) -> str:
    """Produce a structured REFLECT diagnosis for a failed Implementer reply.

    Parameters
    ----------
    response_text : str
        The raw text returned by the Implementer when no parseable
        ``## Report`` block was found.

    Returns
    -------
    str
        A string starting with ``REFLECT:`` followed by a one-paragraph
        verbal diagnosis of the most likely failure cause, a newline, and
        the first 500 characters of ``response_text``.

    Notes
    -----
    Diagnosis categories (checked in order):
    1. Response shorter than 100 characters — too vague or unactionable.
    2. Response contains a capability-limit phrase — outside tool set.
    3. Response has a ``## Report`` heading but missing required subsections.
    4. Response is >= 100 chars with no ``## Report`` heading at all.
    5. Default fallback — malformed for an unrecognised reason.
    """
    truncated = response_text[:500]
    lower = response_text.lower()

    if len(response_text) < 100:
        diagnosis = REFLECT_DIAGNOSIS_SHORT
        return f"REFLECT: {diagnosis}\n{truncated}"

    for phrase in _CAPABILITY_PHRASES:
        if phrase in lower:
            diagnosis = REFLECT_DIAGNOSIS_CAPABILITY_LIMIT
            return f"REFLECT: {diagnosis}\n{truncated}"

    has_report_heading = bool(
        re.search(r"##\s+Report\b", response_text, re.IGNORECASE)
    )

    if has_report_heading:
        missing: list[str] = []
        for sub in _REQUIRED_SUBSECTIONS:
            pattern = rf"###\s+{re.escape(sub)}"
            if not re.search(pattern, response_text, re.IGNORECASE):
                missing.append(sub)
        if missing:
            missing_str = ", ".join(f"'{s}'" for s in missing)
            diagnosis = (
                REFLECT_DIAGNOSIS_MISSING_SUBSECTIONS_TEMPLATE.format(
                    missing_subsections=missing_str
                )
            )
            return f"REFLECT: {diagnosis}\n{truncated}"

    if not has_report_heading:
        diagnosis = REFLECT_DIAGNOSIS_NO_REPORT_HEADING
        return f"REFLECT: {diagnosis}\n{truncated}"

    diagnosis = REFLECT_DIAGNOSIS_DEFAULT
    return f"REFLECT: {diagnosis}\n{truncated}"


# ---------------------------------------------------------------------------
# Transcript helpers (Piece D)
# ---------------------------------------------------------------------------


def _record_transcript(
    transcript_path: Path,
    event: dict[str, Any],
) -> None:
    """Append a single JSONL event to a transcript file.

    Parameters
    ----------
    transcript_path : Path
        Absolute path to the ``.jsonl`` file.  Parent directories are
        created if they do not exist.
    event : dict
        A dict that must include at least a ``"type"`` key and a ``"ts"``
        key with an ISO-format timestamp.
    """
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    with transcript_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(event) + "\n")


def read_transcript(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL transcript file into a list of event dicts.

    Parameters
    ----------
    path : Path
        Absolute path to the ``.jsonl`` transcript file.

    Returns
    -------
    list[dict]
        Ordered list of event dicts as written by ``_record_transcript``.
        Returns an empty list if the file does not exist.
    """
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


class AgenticRun:
    """Orchestrate a Strategizer/Implementer agentic run.

    Parameters
    ----------
    study_dir : Path
        Root of the study tree.  Must contain ``PROBLEM_STATEMENT.md``.
    model : str or None
        LLM model identifier used by both sessions.  If ``None``,
        the backend's ``default_model`` is used.
    checkpoint_every : int
        Number of Implementer delegations between checkpoints.
    stdin : TextIO
        Readable stream for user input (defaults to ``sys.stdin``).
    stdout : TextIO
        Writable stream for user-facing output (defaults to
        ``sys.stdout``).
    backend : Backend or None
        The LLM backend bundle.  If ``None``, defaults to
        :data:`~f3dasm._src.agentic.backends.claude.CLAUDE_BACKEND`.
    strategizer_factory : callable or None
        Override the backend's Strategizer factory.  Signature:
        ``(*, system_prompt, model, tool_closures) ->
        StrategizerSession``.  When supplied, takes precedence over
        the backend's factory (used for test injection).
    implementer_factory : callable or None
        Override the backend's Implementer factory.  Signature:
        ``(*, system_prompt, model, study_dir) ->
        ImplementerSession``.  When supplied, takes precedence over
        the backend's factory (used for test injection).
    study_config : StudyConfig or None
        Config loaded from ``config.yaml``.  CLI kwargs (``model``,
        ``checkpoint_every``) take precedence over config values;
        config values take precedence over backend defaults.
    roles : list[AgentRole] or None
        Named agent roles for the topology.  When ``None`` (the default),
        the runtime synthesises the classic Strategizer + Implementer
        pair using the backend's factories.  Supply a list to define a
        custom agent graph; each role's ``factory`` is auto-detected as
        planner-type (has ``tool_closures`` param) or executor-type (has
        ``study_dir`` param).
    """

    def __init__(
        self,
        study_dir: Path,
        *,
        model: str | None = None,
        checkpoint_every: int = CHECKPOINT_EVERY,
        stdin: TextIO = sys.stdin,
        stdout: TextIO = sys.stdout,
        backend: Backend | None = None,
        strategizer_factory: (
            Callable[..., StrategizerSession] | None
        ) = None,
        implementer_factory: (
            Callable[..., ImplementerSession] | None
        ) = None,
        record_transcripts: bool = True,
        study_config: StudyConfig | None = None,
        topology: Callable[["RunContext"], None] | None = None,
        roles: list[AgentRole] | None = None,
        graph: Graph | None = None,
    ) -> None:
        # Resolve backend: explicit kwarg wins; otherwise honour StudyConfig.backend.
        _cfg_pre = study_config or StudyConfig()
        if backend is not None:
            self._backend: Backend = backend
        else:
            _bname = _cfg_pre.backend
            if _bname == "claude":
                from .backends.claude import CLAUDE_BACKEND as _cb
                self._backend = _cb
            elif _bname == "ollama":
                from .backends.ollama import OLLAMA_BACKEND as _ob
                self._backend = _ob
            elif _bname in _BACKEND_REGISTRY:
                self._backend = _BACKEND_REGISTRY[_bname]
            else:
                raise AgenticRunError(
                    f"Unknown backend {_bname!r}. "
                    "Built-in backends: 'claude', 'ollama'. "
                    "Register custom backends with register_backend()."
                )
        self._study_dir = Path(study_dir).resolve()
        _cfg = _cfg_pre
        # CLI model wins over config, config wins over backend default.
        if model is None:
            model = _cfg.model  # may still be None
        self._model: str = (
            model if model is not None else self._backend.default_model
        )
        # CLI checkpoint_every (explicit) wins over config.
        if checkpoint_every == CHECKPOINT_EVERY and _cfg.checkpoint_every is not None:
            checkpoint_every = _cfg.checkpoint_every
        self._checkpoint_every = checkpoint_every
        # Budget.
        self._budget: timedelta | None = _cfg.budget
        # Custom topology (optional).
        self._topology = topology
        self._start_time: datetime | None = None  # set in execute()
        self._stdin = stdin
        self._stdout = stdout
        # Resolution order for factories: explicit kwarg > backend factory.
        self._strategizer_factory: Callable[..., StrategizerSession] = (
            strategizer_factory
            if strategizer_factory is not None
            else self._backend.strategizer_factory
        )
        self._implementer_factory: Callable[..., ImplementerSession] = (
            implementer_factory
            if implementer_factory is not None
            else self._backend.implementer_factory
        )
        self._record_transcripts = record_transcripts
        # Named-role list (None → synthesise default pair at session start).
        self._roles: list[AgentRole] | None = roles
        # Agent graph (takes priority over roles= when set).
        self._graph: Graph | None = graph

        # Mutable run state — populated in execute().
        self._run_dir: Path | None = None
        self._git_dir: Path | None = None
        self._delegation_counter: int = 0
        self._total_delegations: int = 0
        self._last_report: Report | None = None
        self._done_summary: str | None = None
        self._done_called: bool = False
        self._agents: dict[str, AgentSession] = {}
        self._turn_count: int = 0
        self._checkpoint_summary: str = ""
        self._logger: logging.Logger = logging.getLogger(
            "f3dasm.agentic"
        )

    # ------------------------------------------------------------------
    # Backward-compat properties for _strategizer / _implementer
    # ------------------------------------------------------------------

    @property
    def _strategizer(self) -> AgentSession | None:
        return self._agents.get("strategizer")

    @_strategizer.setter
    def _strategizer(self, v: AgentSession | None) -> None:
        if v is not None:
            self._agents["strategizer"] = v

    @property
    def _implementer(self) -> AgentSession | None:
        return self._agents.get("implementer")

    @_implementer.setter
    def _implementer(self, v: AgentSession | None) -> None:
        if v is not None:
            self._agents["implementer"] = v

    # ------------------------------------------------------------------
    # Budget helpers
    # ------------------------------------------------------------------

    def _remaining(self) -> timedelta | None:
        """Return wall-clock time left, or None if no budget is set.

        Returns
        -------
        timedelta or None
            Remaining time. Can be negative (budget exceeded).
        """
        if self._budget is None or self._start_time is None:
            return None
        return self._budget - (
            datetime.now(tz=timezone.utc) - self._start_time
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def execute(self) -> Path:
        """Run the agentic loop end to end.

        Returns
        -------
        Path
            Absolute path to the ``deliverable/`` folder.

        Raises
        ------
        AgenticRunError
            If ``<study_dir>/PROBLEM_STATEMENT.md`` is missing or a fatal SDK
            error occurs during the run.
        """
        self._validate()
        self._setup_paths()
        self._init_logging()
        self._logger.info(
            "Run starting at %s (model=%s, checkpoint_every=%d)",
            self._run_dir,
            self._model,
            self._checkpoint_every,
        )
        self._init_git()
        self._start_time = datetime.now(tz=timezone.utc)
        briefing_text = (self._study_dir / "PROBLEM_STATEMENT.md").read_text()
        self._logger.info(
            "Read PROBLEM_STATEMENT.md (%d chars)", len(briefing_text)
        )
        self._start_sessions()
        self._logger.info(
            "Sessions started; handing briefing to Strategizer"
        )
        if self._topology is not None:
            self._topology(self._as_context())
        else:
            self._run_loop(briefing_text)
        deliverable = self._assemble_deliverable()
        self._logger.info("Deliverable assembled at %s", deliverable)
        return deliverable

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _init_logging(self) -> None:
        """Configure the f3dasm.agentic logger for this run.

        Adds two handlers — a stderr StreamHandler and a FileHandler
        at ``<run_dir>/run.log`` — both formatted with HH:MM:SS
        timestamps.  Handlers are reset between runs so the same
        AgenticRun-class invocations in one process don't accumulate
        duplicates.
        """
        self._logger = logging.getLogger("f3dasm.agentic")
        self._logger.setLevel(logging.INFO)
        self._logger.handlers.clear()
        self._logger.propagate = False

        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(message)s",
            datefmt="%H:%M:%S",
        )

        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(fmt)
        self._logger.addHandler(stream)

        if self._run_dir is not None:
            log_path = self._run_dir / "run.log"
            file_handler = logging.FileHandler(
                log_path, encoding="utf-8"
            )
            file_handler.setFormatter(fmt)
            self._logger.addHandler(file_handler)

    # ------------------------------------------------------------------
    # Step 1 — Validation
    # ------------------------------------------------------------------

    def _validate(self) -> None:
        briefing = self._study_dir / "PROBLEM_STATEMENT.md"
        if not briefing.exists():
            raise AgenticRunError(
                f"PROBLEM_STATEMENT.md not found in {self._study_dir}. "
                "Create it before running the agentic loop."
            )
        self._preflight()

    def _preflight(self) -> None:
        """Delegate backend preflight check, skipping for stub factories.

        When custom roles are supplied (``roles=`` kwarg), each role
        brings its own factory and the backend preflight is skipped.
        When either factory has been replaced by a test-injected stub,
        we also skip so unit tests do not require the real CLI binary.
        """
        if self._graph is not None or self._roles is not None:
            return  # custom graph/roles: factories come from AgentRole.factory
        using_backend_strategizer = (
            self._strategizer_factory
            is self._backend.strategizer_factory
        )
        using_backend_implementer = (
            self._implementer_factory
            is self._backend.implementer_factory
        )
        if not (using_backend_strategizer or using_backend_implementer):
            return
        if self._backend.name == "ollama":
            from .backends.ollama import _preflight_ollama
            _preflight_ollama(self._model)
            return
        self._backend.preflight()

    # Backward-compatible alias kept so existing code that calls
    # ``run._preflight_check_claude_cli()`` directly still works.
    _preflight_check_claude_cli = _preflight

    # ------------------------------------------------------------------
    # Step 2 — Path setup
    # ------------------------------------------------------------------

    def _setup_paths(self) -> None:
        ts = (
            datetime.now(tz=timezone.utc)
            .isoformat(timespec="seconds")
            .replace(":", "")
        )
        self._run_dir = self._study_dir / "runs" / ts
        self._git_dir = self._run_dir / ".git"
        (self._run_dir / "strategizer_notes").mkdir(parents=True)
        (self._study_dir / "workspace").mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Step 3 — Git init
    # ------------------------------------------------------------------

    def _init_git(self) -> None:
        assert self._run_dir is not None
        assert self._git_dir is not None
        subprocess.run(
            ["git", "init", "--bare", str(self._git_dir)],
            capture_output=True,
            text=True,
            check=True,
        )
        _run_git(
            ["add", "-A"],
            git_dir=self._git_dir,
            work_tree=self._study_dir,
        )
        _run_git(
            [
                "commit",
                "--allow-empty",
                "-m",
                "initial commit (briefing only)",
                "--author=agentic-f3dasm <noreply@f3dasm>",
            ],
            git_dir=self._git_dir,
            work_tree=self._study_dir,
        )

    # ------------------------------------------------------------------
    # Step 4 — Session construction
    # ------------------------------------------------------------------

    def _start_sessions(self) -> None:
        for role in self._resolve_roles():
            self._agents[role.name] = self._instantiate_role(role)

    def _resolve_roles(self) -> list[AgentRole]:
        """Return the effective role list.

        Priority: ``graph=`` > ``roles=`` > default Strategizer + Implementer pair.
        """
        if self._graph is not None:
            return list(self._graph.roles)
        if self._roles is not None:
            return self._roles
        return [
            AgentRole(
                "strategizer",
                factory=self._strategizer_factory,
                reset_on_checkpoint=False,
            ),
            AgentRole(
                "implementer",
                factory=self._implementer_factory,
                reset_on_checkpoint=True,
            ),
        ]

    def _role_by_name(self, name: str) -> AgentRole | None:
        """Return the :class:`AgentRole` with *name*, or ``None``."""
        return next(
            (r for r in self._resolve_roles() if r.name == name), None
        )

    def _instantiate_role(self, role: AgentRole) -> AgentSession:
        """Construct a session for *role*, auto-detecting factory type.

        Planner-type factories (have a ``tool_closures`` parameter) receive
        orchestrator-injected tool closures.  Executor-type factories (have
        a ``study_dir`` parameter) receive the study directory path and use
        the backend's built-in file/bash tools.

        ``_outgoing_targets`` uses :meth:`_resolve_roles` (not
        ``self._agents``) so it is safe to call here before the full
        ``_agents`` dict is populated.
        """
        sig = inspect.signature(role.factory)
        prompt = role.system_prompt or self._default_prompt_for(role.name)
        if "tool_closures" in sig.parameters:
            closures, _ = self._build_planner_tools(role.name)
            return role.factory(
                system_prompt=prompt,
                model=self._model,
                tool_closures=closures,
                **role.kwargs,
            )
        return role.factory(
            system_prompt=prompt,
            model=self._model,
            study_dir=self._study_dir,
            **role.kwargs,
        )

    def _default_prompt_for(self, name: str) -> str:
        """Return the default system prompt for a named role.

        Planner-type roles (factory has ``tool_closures`` param) get the
        Strategizer prompt; all others get the Implementer prompt.
        """
        role = self._role_by_name(name)
        if role is not None:
            sig = inspect.signature(role.factory)
            if "tool_closures" in sig.parameters:
                return self._compose_strategizer_prompt(name)
        elif name == "strategizer":
            return self._compose_strategizer_prompt(name)
        return self._compose_implementer_prompt()

    def _reset_implementer(self, checkpoint_summary: str) -> None:
        """Destroy the current Implementer and create a fresh one."""
        self._checkpoint_summary = checkpoint_summary
        self._reset_agent("implementer")

    def _reset_agent(self, name: str) -> None:
        """Destroy the session for *name* and create a fresh one."""
        role = self._role_by_name(name)
        if role is None:
            return
        self._logger.info("Agent %r session reset (briefed with checkpoint summary)", name)
        reset_msg = IMPLEMENTER_RESET_PROMPT_TEMPLATE.format(
            checkpoint_summary=self._checkpoint_summary
        )
        self._agents[name] = self._instantiate_role(role)
        self._agents[name].send(reset_msg)

    def _compose_strategizer_prompt(
        self, name: str = "strategizer"
    ) -> str:
        """Return the planner system prompt with run-specific paths and roster.

        Prepends a small ``<run_paths>`` section so the model knows the
        canonical absolute paths for its notes directory and the study
        tree.  Also injects ``<available_agents>`` listing only the agents
        this particular planner can reach via ``Delegate``.
        """
        assert self._run_dir is not None
        notes_dir = (self._run_dir / "strategizer_notes").resolve()
        preamble = RUN_PATHS_PREAMBLE_TEMPLATE.format(
            study_dir=self._study_dir,
            notes_dir=notes_dir,
        )
        prompt = preamble + STRATEGIZER_SYSTEM_PROMPT
        outgoing = self._outgoing_targets(name)
        if outgoing:
            peer_roles = [
                r for r in self._resolve_roles() if r.name in outgoing
            ]
            lines = [
                f'- {r.name}: {r.description or "agent"}'
                for r in peer_roles
            ]
            roster = "\n".join(lines)
            prompt += (
                "\n\n<available_agents>\n"
                "Agents you can reach via "
                'Delegate(target="<name>", intent=..., expected_report=...):\n'
                f"{roster}\n"
                "</available_agents>"
            )
        return prompt

    def _compose_implementer_prompt(self) -> str:
        """Return the Implementer system prompt with workspace anchored.

        Prepends a ``<workspace>`` section so the model knows the
        absolute path it must write to. The session's cwd is the study
        directory, so relative paths still resolve correctly; the
        explicit absolute path here is what removes the temptation to
        use ``/tmp``.
        """
        from .agent_prompts import IMPLEMENTER_SYSTEM_PROMPT_OLLAMA
        workspace = (self._study_dir / "workspace").resolve()
        preamble = WORKSPACE_PREAMBLE_TEMPLATE.format(
            workspace_dir=workspace,
        )
        base = (
            IMPLEMENTER_SYSTEM_PROMPT_OLLAMA
            if self._backend.name == "ollama"
            else IMPLEMENTER_SYSTEM_PROMPT
        )
        return preamble + base

    # ------------------------------------------------------------------
    # Step 5 — Main loop
    # ------------------------------------------------------------------

    def _run_loop(self, briefing_text: str) -> None:
        """Drive the entry agent until Done() is called or a stop occurs."""
        entry_name = (
            self._graph.entry if self._graph is not None else "strategizer"
        )
        entry_agent = self._agents.get(entry_name)
        assert entry_agent is not None, (
            f"Entry agent {entry_name!r} not found in _agents"
        )
        entry_agent.send(briefing_text)
        self._turn_count += 1

    # ------------------------------------------------------------------
    # Tool closures (Strategizer)
    # ------------------------------------------------------------------

    def _outgoing_targets(self, agent_name: str) -> list[str]:
        """Return the list of agent names *agent_name* can delegate to.

        When a :class:`~backends.base.Graph` is set, reads from its edge
        declarations.  Otherwise returns all other agent names (legacy
        all-to-all behaviour), computed from :meth:`_resolve_roles` so
        it is safe to call before ``_agents`` is fully populated.
        """
        if self._graph is not None:
            return [
                e.target
                for e in self._graph.edges
                if e.source == agent_name
            ]
        all_names = [r.name for r in self._resolve_roles()]
        return [n for n in all_names if n != agent_name]

    def _build_planner_tools(
        self,
        agent_name: str = "strategizer",
    ) -> tuple[dict[str, Callable[..., str]], list[str]]:
        """Return ``(closures, outgoing)`` for *agent_name*.

        ``closures`` contains only the tools the agent is permitted to
        use (filtered by :attr:`AgentRole.tools` and outgoing edges).
        ``outgoing`` is the list of agent names the agent can delegate
        to, used to build per-agent Ollama tool schemas.

        When ``record_transcripts`` is True each closure is wrapped so
        that every call and result is appended to
        ``runs/<ts>/transcripts/<agent_name>.jsonl``.
        """
        role = self._role_by_name(agent_name)
        allowed = role.tools if role is not None else None
        outgoing = self._outgoing_targets(agent_name)

        raw: dict[str, Callable[..., str]] = {
            "Read": self._tool_read,
            "WriteMarkdown": self._tool_write_markdown,
            "Ask": self._tool_ask,
            "Done": self._tool_done,
        }
        if outgoing:
            _out = list(outgoing)
            _caller = agent_name
            _default_target = _out[0]

            def _delegate_scoped(
                intent: str,
                expected_report: str,
                target: str = _default_target,
                _o: list[str] = _out,
                _c: str = _caller,
            ) -> str:
                """Delegate a task — only to permitted targets."""
                if target not in _o:
                    return (
                        f"ERROR: {_c!r} is not permitted to delegate to "
                        f"{target!r}. Permitted targets: {_o}"
                    )
                return self._tool_delegate(
                    intent, expected_report, target, caller=_c
                )

            raw["Delegate"] = _delegate_scoped

        if allowed is not None:
            raw = {k: v for k, v in raw.items() if k in allowed}

        if not self._record_transcripts:
            return raw, outgoing
        wrapped = {
            name: self._make_transcript_wrapper(name, fn, agent_name)
            for name, fn in raw.items()
        }
        return wrapped, outgoing

    # Keep the old name as an alias so any external callers still work.
    _build_strategizer_tools = _build_planner_tools

    def _make_transcript_wrapper(
        self,
        tool_name: str,
        fn: Callable[..., str],
        agent_name: str = "strategizer",
    ) -> Callable[..., str]:
        """Return a wrapped version of *fn* that records to <agent_name>.jsonl."""

        original_sig = inspect.signature(fn)
        _aname = agent_name

        def _wrapper(**kwargs: Any) -> str:
            ts_call = (
                datetime.now(tz=timezone.utc)
                .isoformat(timespec="seconds")
            )
            if self._run_dir is not None:
                t_path = (
                    self._run_dir / "transcripts" / f"{_aname}.jsonl"
                )
                _record_transcript(
                    t_path,
                    {
                        "type": "tool_call",
                        "name": tool_name,
                        "args": kwargs,
                        "ts": ts_call,
                    },
                )
            result = fn(**kwargs)
            ts_result = (
                datetime.now(tz=timezone.utc)
                .isoformat(timespec="seconds")
            )
            if self._run_dir is not None:
                _record_transcript(
                    self._run_dir / "transcripts" / f"{_aname}.jsonl",
                    {
                        "type": "tool_result",
                        "name": tool_name,
                        "content": result,
                        "ts": ts_result,
                    },
                )
            return result

        _wrapper.__signature__ = original_sig
        _wrapper.__doc__ = fn.__doc__
        _wrapper.__name__ = getattr(fn, "__name__", tool_name)
        return _wrapper

    def _tool_read(self, path: str) -> str:
        """Read a file from the study tree.

        Returns the file contents as text, or an ERROR string if the
        path is rejected.
        """
        target = (self._study_dir / path).resolve()
        if not str(target).startswith(str(self._study_dir)):
            return (
                f"ERROR: path {path!r} escapes the study directory. "
                "Only files under the study root may be read."
            )
        if not target.exists():
            return f"ERROR: file not found: {path}"
        try:
            return target.read_text()
        except Exception as exc:
            return f"ERROR: could not read {path}: {exc}"

    def _tool_write_markdown(self, path: str, content: str) -> str:
        """Write a Markdown file to ``strategizer_notes/``.

        Relative paths and bare filenames are resolved under
        ``<run_dir>/strategizer_notes/``.  Absolute paths that already
        live under that directory are accepted as-is.  Anything that
        escapes the notes directory or carries a non-``.md`` extension
        is rejected.

        Returns OK or an ERROR string.
        """
        assert self._run_dir is not None
        notes_dir = (self._run_dir / "strategizer_notes").resolve()

        candidate = Path(path)
        if candidate.is_absolute():
            target = candidate.resolve()
        else:
            # Strip any leading "strategizer_notes/" or "runs/<ts>/..."
            # prefix the model might have invented; we always anchor the
            # write under the canonical notes directory.
            stripped = path
            for prefix in ("./", "strategizer_notes/"):
                if stripped.startswith(prefix):
                    stripped = stripped[len(prefix):]
            target = (notes_dir / stripped).resolve()

        if not target.suffix == ".md":
            return (
                f"ERROR: WriteMarkdown only accepts .md files; "
                f"got {path!r}."
            )

        try:
            target.relative_to(notes_dir)
        except ValueError:
            return (
                f"ERROR: WriteMarkdown path escapes "
                f"<run_dir>/strategizer_notes/. Got {path!r}."
            )

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return f"OK: wrote {target}"

    def _tool_ask(self, question: str) -> str:
        """Print a question to stdout and block on stdin.

        Returns the user's typed line as the tool result.
        """
        self._stdout.write(f"\n[Strategizer asks]: {question}\n> ")
        self._stdout.flush()
        line = self._stdin.readline()
        self._turn_count += 1
        return line.rstrip("\n") or "(no reply)"

    def _tool_delegate(
        self,
        intent: str,
        expected_report: str,
        target: str = "implementer",
        *,
        caller: str = "strategizer",
    ) -> str:
        """Delegate a task to a named agent.

        Records a git commit, sends the task to *target*, parses the
        Report, and returns the raw Report markdown to the caller.  If
        no parseable Report is found, returns an ERROR without
        incrementing the delegation counter.

        Parameters
        ----------
        intent : str
            What the agent should do.
        expected_report : str
            What measurements or conclusions are required.
        target : str
            Name of the agent to delegate to (default: ``"implementer"``).
            Must be a key in ``self._agents``.

        Returns
        -------
        str
            The agent's ``## Report`` block, or an ERROR string.
        """
        # Route to the named agent.
        agent = self._agents.get(target)
        if agent is None:
            return (
                f"ERROR: no agent named {target!r}. "
                f"Available agents: {list(self._agents)}"
            )

        # Budget check — before any work for this delegation.
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

        assert self._git_dir is not None

        deleg_idx = self._total_delegations
        delegation_start = time.monotonic()
        self._logger.info(
            "[delegation #%d] starting — intent: %s",
            deleg_idx,
            _preview(intent, 80),
        )

        # (a) Git commit — record study state prior to this delegation.
        if self._last_report is not None:
            commit_msg = _truncate_for_commit(
                self._last_report.conclusions
                or self._last_report.actions_taken
                or "implementer run"
            )
        else:
            commit_msg = "initial commit"

        _run_git(
            ["add", "-A"],
            git_dir=self._git_dir,
            work_tree=self._study_dir,
            check=False,
        )
        _run_git(
            [
                "commit",
                "--allow-empty",
                "-m",
                commit_msg,
                "--author=agentic-f3dasm <noreply@f3dasm>",
            ],
            git_dir=self._git_dir,
            work_tree=self._study_dir,
            check=False,
        )

        # (b) Build task message.
        remaining = self._remaining()
        task = Task(
            intent=intent,
            expected_report=expected_report,
            remaining_time=remaining,
            budget=self._budget,
        )
        task_msg = _format_task(task)

        # (c) Send to target agent; record per-delegation transcript.
        impl_reply = agent.send(task_msg)
        self._turn_count += 1

        if self._record_transcripts and self._run_dir is not None:
            ts_now = (
                datetime.now(tz=timezone.utc)
                .isoformat(timespec="seconds")
            )
            deleg_idx = self._total_delegations
            impl_transcript = (
                self._run_dir
                / "transcripts"
                / f"{deleg_idx}_{target}.jsonl"
            )
            _record_transcript(
                impl_transcript,
                {
                    "type": "user_message",
                    "content": task_msg,
                    "ts": ts_now,
                },
            )
            _record_transcript(
                impl_transcript,
                {
                    "type": "assistant_text",
                    "content": impl_reply,
                    "ts": ts_now,
                },
            )

        # (d) Parse Report.  If the first reply is malformed, give the
        # Implementer one focused corrective retry before falling through
        # to REFLECT.  The retry message restates the required structure
        # in literal form so the model cannot misread it.
        report = _parse_report(impl_reply)
        if report is None:
            correction = IMPLEMENTER_REPORT_RETRY_PROMPT
            retry_reply = agent.send(correction)
            self._turn_count += 1
            if self._record_transcripts and self._run_dir is not None:
                ts_now = (
                    datetime.now(tz=timezone.utc)
                    .isoformat(timespec="seconds")
                )
                _record_transcript(
                    impl_transcript,
                    {
                        "type": "user_message",
                        "content": correction,
                        "ts": ts_now,
                    },
                )
                _record_transcript(
                    impl_transcript,
                    {
                        "type": "assistant_text",
                        "content": retry_reply,
                        "ts": ts_now,
                    },
                )
            impl_reply = retry_reply
            report = _parse_report(impl_reply)
        if report is None:
            reflect_msg = _classify_failed_implementer_response(
                impl_reply
            )
            if self._record_transcripts and self._run_dir is not None:
                ts_now = (
                    datetime.now(tz=timezone.utc)
                    .isoformat(timespec="seconds")
                )
                caller_transcript = (
                    self._run_dir
                    / "transcripts"
                    / f"{caller}.jsonl"
                )
                _record_transcript(
                    caller_transcript,
                    {
                        "type": "tool_result",
                        "name": "Delegate",
                        "content": reflect_msg,
                        "ts": ts_now,
                    },
                )
            elapsed = _format_elapsed(
                time.monotonic() - delegation_start
            )
            self._logger.warning(
                "[delegation #%d] failed after retry in %s — REFLECT: %s",
                deleg_idx,
                elapsed,
                _preview(
                    reflect_msg.split("\n", 1)[0].removeprefix(
                        "REFLECT: "
                    ),
                    100,
                ),
            )
            return reflect_msg

        # (e) On success — update state.
        self._delegation_counter += 1
        self._total_delegations += 1
        self._last_report = report
        report.remaining_time = self._remaining()
        elapsed = _format_elapsed(time.monotonic() - delegation_start)
        self._logger.info(
            "[delegation #%d] done in %s — conclusions: %s",
            deleg_idx,
            elapsed,
            _preview(report.conclusions, 100),
        )

        # Check checkpoint.
        if self._delegation_counter >= self._checkpoint_every:
            self._run_checkpoint()

        return report.raw

    def _tool_done(self, summary: str) -> str:
        """Signal end of run with the Strategizer's final summary.

        Sets the ``_done_called`` flag so the main loop exits.

        Parameters
        ----------
        summary : str
            The Strategizer's scientific conclusion.

        Returns
        -------
        str
            Acknowledgement string.
        """
        self._done_summary = summary
        self._done_called = True
        self._logger.info(
            "Done received from Strategizer — summary: %s",
            _preview(summary, 120),
        )
        return "OK: run finalised. Assembling deliverable."

    # ------------------------------------------------------------------
    # Topology helpers
    # ------------------------------------------------------------------

    def _delegate_task(
        self, task: Task, target: str = "implementer"
    ) -> Delegation:
        """Delegate *task* to *target*; return a completed Delegation.

        Used by custom topology callables via :class:`_RunContextImpl`.
        """
        raw = self._tool_delegate(
            intent=task.intent,
            expected_report=task.expected_report,
            target=target,
        )
        d = Delegation(task=task)
        if _parse_delegation(raw, d) is None:
            d.report = Report(
                actions_taken="",
                files_touched=[],
                conclusions=raw,
                numbers={},
                raw=raw,
            )
            d.metadata["error"] = True
        return d

    def _as_context(self) -> "_RunContextImpl":
        """Return a :class:`_RunContextImpl` wrapping this run."""
        return _RunContextImpl(self)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _run_checkpoint(self) -> None:
        """Inject the checkpoint prompt and handle user steering."""
        entry_name = (
            self._graph.entry if self._graph is not None else "strategizer"
        )
        entry_agent = self._agents.get(entry_name) or self._strategizer
        assert entry_agent is not None

        self._logger.info(
            "Checkpoint firing after %d delegations",
            self._total_delegations,
        )

        self._stdout.write(
            "\n--- CHECKPOINT "
            f"({self._total_delegations} delegations) ---\n"
        )
        self._stdout.flush()

        cp_reply = entry_agent.send(CHECKPOINT_STRATEGIZER_PROMPT)
        self._turn_count += 1
        self._checkpoint_summary = cp_reply

        self._stdout.write("\n" + cp_reply + "\n")
        self._stdout.write(
            '\n[Checkpoint] Type "stop" to end, or press Enter '
            "to continue (optionally with steering):\n> "
        )
        self._stdout.flush()

        user_input = self._stdin.readline().rstrip("\n")

        if user_input.strip() == "stop":
            self._done_called = True
            return

        # Resume: pass user steering (or "(continue)") to the entry agent.
        steering_msg = (
            f"user steering: {user_input}"
            if user_input.strip()
            else "(continue)"
        )
        entry_agent.send(steering_msg)
        self._turn_count += 1

        # Reset agents according to per-role policy.
        for role in self._resolve_roles():
            if role.reset_on_checkpoint:
                self._reset_agent(role.name)

        # Reset delegation counter.
        self._delegation_counter = 0

    # ------------------------------------------------------------------
    # Deliverable assembly
    # ------------------------------------------------------------------

    def _assemble_deliverable(self) -> Path:
        """Write the deliverable folder and return its path."""
        assert self._run_dir is not None
        assert self._git_dir is not None

        deliv_dir = self._run_dir / "deliverable"
        deliv_dir.mkdir(parents=True, exist_ok=True)

        # Determine final answer.
        if self._done_summary:
            final_answer = self._done_summary
        elif self._checkpoint_summary:
            final_answer = self._checkpoint_summary
        else:
            final_answer = (
                "(Run ended without a Done summary or checkpoint.)"
            )

        # Git log (last 30 lines).
        git_log_result = _run_git(
            ["log", "--oneline"],
            git_dir=self._git_dir,
            work_tree=self._study_dir,
            check=False,
        )
        git_log_oneline = git_log_result.stdout.strip()
        last_30 = "\n".join(
            git_log_oneline.splitlines()[-30:]
        )

        git_log_full_result = _run_git(
            [
                "log",
                "--pretty=format:%h %ai %s",
            ],
            git_dir=self._git_dir,
            work_tree=self._study_dir,
            check=False,
        )
        git_log_full = git_log_full_result.stdout.strip()

        # solution.md
        now_ts = datetime.now(tz=timezone.utc).isoformat(
            timespec="seconds"
        )
        solution_lines = [
            "# Solution",
            "",
            final_answer,
            "",
            "## Run metadata",
            "",
            f"- timestamp: {now_ts}",
            f"- model: {self._model}",
            f"- total_delegations: {self._total_delegations}",
            f"- total_turns: {self._turn_count}",
            f"- run_dir: {self._run_dir}",
        ]
        if self._budget is not None:
            solution_lines.append(f"- budget: {self._budget}")
        if self._start_time is not None:
            used = datetime.now(tz=timezone.utc) - self._start_time
            h, rem = divmod(int(used.total_seconds()), 3600)
            m, s = divmod(rem, 60)
            solution_lines.append(f"- time_used: {h:02d}:{m:02d}:{s:02d}")
        solution_lines += [
            "",
            "## Provenance",
            "",
            "```",
            last_30,
            "```",
        ]
        (deliv_dir / "solution.md").write_text(
            "\n".join(solution_lines)
        )

        # git_log.txt
        (deliv_dir / "git_log.txt").write_text(git_log_full)

        # replication/ — copy workspace/ contents.
        replication_dir = deliv_dir / "replication"
        replication_dir.mkdir(exist_ok=True)
        workspace = self._study_dir / "workspace"
        if workspace.exists():
            for item in workspace.iterdir():
                dest = replication_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest)

        # transcripts/ — copy run transcripts to deliverable.
        if self._record_transcripts:
            transcripts_src = self._run_dir / "transcripts"
            if transcripts_src.exists():
                transcripts_dest = deliv_dir / "transcripts"
                shutil.copytree(
                    transcripts_src,
                    transcripts_dest,
                    dirs_exist_ok=True,
                )

        # run.log — flush handlers and copy the log into the deliverable.
        log_src = self._run_dir / "run.log"
        for handler in list(self._logger.handlers):
            handler.flush()
        if log_src.exists():
            shutil.copy2(log_src, deliv_dir / "run.log")

        return deliv_dir


# ---------------------------------------------------------------------------
# RunContext implementation
# ---------------------------------------------------------------------------


class _RunContextImpl:
    """Concrete :class:`RunContext` that wraps an :class:`AgenticRun`.

    Exposes only the stable interface so topology callables do not need
    to depend on :class:`AgenticRun` internals.
    """

    def __init__(self, run: AgenticRun) -> None:
        self._run = run

    def delegate(
        self, task: Task, target: str = "implementer"
    ) -> Delegation:
        """Send *task* to *target*; return the completed Delegation."""
        return self._run._delegate_task(task, target=target)

    def checkpoint(self) -> None:
        """Commit run state to git and reset the delegation counter."""
        self._run._run_checkpoint()

    def budget_remaining(self) -> timedelta | None:
        """Wall-clock time remaining, or ``None`` if no budget is set."""
        return self._run._remaining()

    def done(self, summary: str) -> None:
        """Signal the run is complete with a final summary string."""
        self._run._done_summary = summary
        self._run._done_called = True

    def ask(self, question: str) -> str:
        """Print *question* to stdout and return the operator's reply."""
        return self._run._tool_ask(question=question)

    def parallel(
        self,
        task_fn: "Callable[[int], Task]",
        targets: list[str],
    ) -> list[Delegation]:
        """Sequential-safe fan-out through the run pipeline."""
        results: list[Delegation] = []
        for i, target in enumerate(targets):
            task = task_fn(i)
            try:
                d = self.delegate(task, target=target)
            except Exception as exc:
                d = Delegation(task=task)
                d.report = Report(
                    actions_taken="",
                    files_touched=[],
                    conclusions=f"ERROR: {exc}",
                    numbers={},
                    raw="",
                )
                d.metadata["error"] = True
            results.append(d)
        return results

    _RETRY_CORRECTIVE = (
        "Attempt {attempt} failed. Review what went wrong and revise your approach."
    )

    def retry(
        self,
        task: Task,
        target: str = "implementer",
        *,
        is_success: Callable[[Delegation], bool],
        max_fails: int = 3,
        on_failure: Callable[[Delegation, int], str] | None = None,
    ) -> Delegation:
        """Persistence loop through the run pipeline."""
        fails = 0
        current_task = task
        while True:
            d = self.delegate(current_task, target=target)
            if d.is_complete and is_success(d):
                return d
            fails += 1
            if fails >= max_fails:
                raise AgenticRunError(
                    f"ctx.retry: exceeded max_fails={max_fails} for target={target!r}."
                )
            corrective_text = (
                on_failure(d, fails)
                if on_failure is not None
                else self._RETRY_CORRECTIVE.format(attempt=fails)
            )
            current_task = Task(
                intent=corrective_text,
                expected_report=task.expected_report,
            )

    def debate(
        self,
        target_a: str,
        target_b: str,
        n: int,
        initial: str,
    ) -> list[Delegation]:
        """Fixed-N debate through the run pipeline."""
        if n < 1:
            raise ValueError(f"ctx.debate: n must be ≥ 1, got {n}.")
        transcript: list[Delegation] = []
        current = initial
        for _ in range(n):
            task_a = Task(intent=current, expected_report="Respond to the debate.")
            d_a = self.delegate(task_a, target=target_a)
            transcript.append(d_a)
            a_reply = d_a.report.conclusions if d_a.is_complete else current
            task_b = Task(intent=a_reply, expected_report="Respond to the debate.")
            d_b = self.delegate(task_b, target=target_b)
            transcript.append(d_b)
            current = d_b.report.conclusions if d_b.is_complete else a_reply
        return transcript


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_task(task: Task) -> str:
    """Render a Task as a structured markdown block.

    Parameters
    ----------
    task : Task
        The task to format.

    Returns
    -------
    str
        Markdown string ready for the Implementer's context.
    """
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


def _format_delegation(delegation: Delegation) -> str:
    """Render the task half of *delegation* as a markdown block.

    Delegates to :func:`_format_task` with ``delegation.task``.

    Parameters
    ----------
    delegation : Delegation
        The envelope whose task fields are to be formatted.

    Returns
    -------
    str
        Markdown string ready for the responding agent's context.
    """
    return _format_task(delegation.task)


def _parse_delegation(
    text: str, delegation: Delegation
) -> Delegation | None:
    """Parse an agent reply into ``delegation.report``.

    Calls :func:`_parse_report` on *text*.  Returns ``None`` if no
    ``## Report`` heading is found.  On success sets
    ``delegation.report`` to the parsed :class:`Report` and returns the
    enriched *delegation* (same object).

    Parameters
    ----------
    text : str
        Full assistant message text.
    delegation : Delegation
        The envelope to enrich with the parsed report.

    Returns
    -------
    Delegation or None
        The enriched *delegation*, or ``None`` if parsing failed.
    """
    report = _parse_report(text)
    if report is None:
        return None
    delegation.report = report
    return delegation


