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
import asyncio
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, TextIO, runtime_checkable

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
    "CHECKPOINT_EVERY",
    "Report",
    "Task",
    "read_transcript",
]

# ---------------------------------------------------------------------------
# Module constant
# ---------------------------------------------------------------------------

CHECKPOINT_EVERY: int = 30
"""Number of Implementer delegations between checkpoints."""

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
    """

    intent: str
    expected_report: str


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
    """

    actions_taken: str
    files_touched: list[str]
    conclusions: str
    numbers: dict[str, Any]
    raw: str


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class AgenticRunError(RuntimeError):
    """Raised for non-recoverable orchestrator failures."""


# ---------------------------------------------------------------------------
# Session protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class StrategizerSession(Protocol):
    """Minimal protocol for a Strategizer session.

    The orchestrator calls ``send(message)`` and receives the full text
    of the assistant reply.  The session is responsible for maintaining
    conversation history across calls.
    """

    def send(self, message: str) -> str:
        """Send a user message; return the assistant reply."""
        ...


@runtime_checkable
class ImplementerSession(Protocol):
    """Minimal protocol for an Implementer session."""

    def send(self, message: str) -> str:
        """Send a user message; return the assistant reply."""
        ...


# ---------------------------------------------------------------------------
# Concrete Claude SDK sessions
# ---------------------------------------------------------------------------


class _ClaudeStrategizer:
    """Claude Agent SDK-backed Strategizer session.

    The Strategizer receives five custom tools (Read, WriteMarkdown, Ask,
    Delegate, Done) via an in-process MCP server built with
    ``create_sdk_mcp_server``.  The SDK handles tool invocation
    automatically; the orchestrator supplies each tool's closure via the
    ``tool_closures`` argument so that the closures can mutate orchestrator
    state.

    The session is resumed across ``send()`` calls via the
    ``resume=session_id`` mechanism so conversation history accumulates
    in the CLI process.

    Parameters
    ----------
    system_prompt : str
        Strategizer system prompt.
    model : str
        Claude model identifier.
    tool_closures : dict[str, Callable[..., str]]
        Mapping of tool name to synchronous Python callable.  Each
        callable must accept the keyword arguments that match the tool's
        JSON schema.
    """

    def __init__(
        self,
        system_prompt: str,
        model: str,
        tool_closures: dict[str, Callable[..., str]],
    ) -> None:
        self._system_prompt = system_prompt
        self._model = model
        self._tool_closures = tool_closures
        self._session_id: str | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_options(self) -> Any:
        """Build ClaudeAgentOptions for the next query() call."""
        from claude_agent_sdk import (
            ClaudeAgentOptions,
            SdkMcpTool,
            create_sdk_mcp_server,
        )

        sdk_tools: list[SdkMcpTool] = []
        for tool_name, fn in self._tool_closures.items():
            schema = _infer_schema_from_callable(fn)

            def _make_handler(
                bound_fn: Callable[..., str],
            ) -> Callable[[dict[str, Any]], Any]:
                async def _handler(
                    args: dict[str, Any],
                ) -> dict[str, Any]:
                    try:
                        result = bound_fn(**args)
                    except Exception as exc:
                        return {
                            "content": [
                                {
                                    "type": "text",
                                    "text": (
                                        f"ERROR: {type(exc).__name__}: {exc}"
                                    ),
                                }
                            ],
                            "is_error": True,
                        }
                    text = str(result) if result is not None else ""
                    return {
                        "content": [{"type": "text", "text": text}]
                    }

                return _handler

            sdk_tools.append(
                SdkMcpTool(
                    name=tool_name,
                    description=(
                        (fn.__doc__ or tool_name).split("\n")[0].strip()
                    ),
                    input_schema=schema,
                    handler=_make_handler(fn),
                )
            )

        server_name = "f3dasm_strategizer_tools"
        mcp_cfg = create_sdk_mcp_server(
            name=server_name,
            tools=sdk_tools if sdk_tools else None,
        )
        qualified_tool_names = [
            f"mcp__{server_name}__{t.name}" for t in sdk_tools
        ]

        return ClaudeAgentOptions(
            system_prompt=self._system_prompt,
            model=self._model,
            tools=[],
            mcp_servers={server_name: mcp_cfg},
            allowed_tools=qualified_tool_names,
            disallowed_tools=_IMPLEMENTER_DENY_LIST,
            permission_mode="bypassPermissions",
            resume=self._session_id,
            strict_mcp_config=True,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def send(self, message: str) -> str:
        """Send a user message and return the final assistant text."""
        from claude_agent_sdk import (
            AssistantMessage,
            ResultMessage,
            TextBlock,
            query,
        )

        options = self._build_options()
        assistant_text = ""
        result_msg: Any = None

        async def _run() -> None:
            nonlocal assistant_text, result_msg
            last_assistant: Any = None
            gen = query(prompt=message, options=options)
            try:
                async for msg in gen:
                    if isinstance(msg, AssistantMessage):
                        last_assistant = msg
                    elif isinstance(msg, ResultMessage):
                        result_msg = msg
                        if last_assistant is not None:
                            for block in last_assistant.content:
                                if isinstance(block, TextBlock):
                                    assistant_text += block.text
                        break
            finally:
                aclose = getattr(gen, "aclose", None)
                if aclose is not None:
                    try:
                        await aclose()
                    except Exception:
                        pass

        try:
            _run_async_safe(_run())
        except AgenticRunError:
            raise
        except Exception as exc:
            try:
                from claude_agent_sdk._errors import ClaudeSDKError
            except ImportError:
                raise
            if isinstance(exc, ClaudeSDKError):
                raise _classify_sdk_error(exc, "Strategizer") from exc
            raise

        if result_msg is not None:
            self._session_id = result_msg.session_id

        if result_msg is not None and result_msg.is_error:
            raise AgenticRunError(
                f"Strategizer SDK error: {result_msg}"
            )
        return assistant_text


class _ClaudeImplementer:
    """Claude Agent SDK-backed Implementer session.

    The Implementer uses the SDK's built-in file/exec tools (Read, Write,
    Edit, Bash, Glob, Grep) restricted to the study directory.  No custom
    MCP tools are registered; the Implementer communicates its results
    via a ``## Report`` block in the final assistant message.

    Parameters
    ----------
    system_prompt : str
        Implementer system prompt.
    model : str
        Claude model identifier.
    study_dir : Path
        The study root.  The SDK's ``cwd`` is set here so relative paths
        in tool calls resolve correctly.
    """

    def __init__(
        self,
        system_prompt: str,
        model: str,
        study_dir: Path,
    ) -> None:
        self._system_prompt = system_prompt
        self._model = model
        self._study_dir = study_dir
        self._session_id: str | None = None

    def send(self, message: str) -> str:
        """Send a user message and return the final assistant text."""
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ResultMessage,
            TextBlock,
            query,
        )

        options = ClaudeAgentOptions(
            system_prompt=self._system_prompt,
            model=self._model,
            cwd=str(self._study_dir),
            tools=_IMPLEMENTER_ALLOWED_TOOLS,
            allowed_tools=_IMPLEMENTER_ALLOWED_TOOLS,
            disallowed_tools=_IMPLEMENTER_DENY_LIST,
            permission_mode="bypassPermissions",
            resume=self._session_id,
        )

        assistant_text = ""
        result_msg: Any = None

        async def _run() -> None:
            nonlocal assistant_text, result_msg
            last_assistant: Any = None
            gen = query(prompt=message, options=options)
            try:
                async for msg in gen:
                    if isinstance(msg, AssistantMessage):
                        last_assistant = msg
                    elif isinstance(msg, ResultMessage):
                        result_msg = msg
                        if last_assistant is not None:
                            for block in last_assistant.content:
                                if isinstance(block, TextBlock):
                                    assistant_text += block.text
                        break
            finally:
                aclose = getattr(gen, "aclose", None)
                if aclose is not None:
                    try:
                        await aclose()
                    except Exception:
                        pass

        try:
            _run_async_safe(_run())
        except AgenticRunError:
            raise
        except Exception as exc:
            try:
                from claude_agent_sdk._errors import ClaudeSDKError
            except ImportError:
                raise
            if isinstance(exc, ClaudeSDKError):
                raise _classify_sdk_error(exc, "Implementer") from exc
            raise

        if result_msg is not None:
            self._session_id = result_msg.session_id

        if result_msg is not None and result_msg.is_error:
            raise AgenticRunError(
                f"Implementer SDK error: {result_msg}"
            )
        return assistant_text


# ---------------------------------------------------------------------------
# Async runner helper
# ---------------------------------------------------------------------------


def _run_async_safe(coro: Any) -> None:
    """Run a coroutine safely whether or not an event loop is running.

    When called from within an existing asyncio event loop (e.g. from a
    tool handler inside a running ``asyncio.run()`` session), running a
    second ``asyncio.run()`` would raise ``RuntimeError: This event loop
    is already running``.  This helper detects that case and runs the
    coroutine in a fresh event loop on a dedicated background thread,
    joining the thread before returning so the call is effectively
    synchronous from the caller's perspective.

    Parameters
    ----------
    coro : coroutine
        The coroutine to run.
    """
    import concurrent.futures

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is not None and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            future = ex.submit(asyncio.run, coro)
            future.result()
    else:
        asyncio.run(coro)


# ---------------------------------------------------------------------------
# SDK tool lists
# ---------------------------------------------------------------------------

_IMPLEMENTER_ALLOWED_TOOLS: list[str] = [
    "Read",
    "Write",
    "Edit",
    "MultiEdit",
    "Bash",
    "Glob",
    "Grep",
]
"""Built-in tools the Implementer may use."""

_IMPLEMENTER_DENY_LIST: list[str] = [
    "WebSearch",
    "WebFetch",
    "Task",
    "ExitPlanMode",
    "computer",
]
"""Tools that are always disallowed for both agents."""

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _infer_schema_from_callable(
    fn: Callable[..., Any],
) -> dict[str, Any]:
    """Derive a minimal JSON Schema from a callable's type annotations.

    Parameters
    ----------
    fn : callable
        The synchronous Python callable to inspect.

    Returns
    -------
    dict
        JSON Schema dict with ``type``, ``properties``, and optionally
        ``required``.
    """
    import inspect

    sig = inspect.signature(fn)
    props: dict[str, Any] = {}
    required: list[str] = []
    _TYPE_MAP = {
        int: {"type": "integer"},
        float: {"type": "number"},
        bool: {"type": "boolean"},
        dict: {"type": "object"},
        list: {"type": "array"},
        str: {"type": "string"},
    }
    for pname, param in sig.parameters.items():
        ann = param.annotation
        if ann is inspect.Parameter.empty:
            json_type: dict[str, Any] = {"type": "string"}
        else:
            json_type = _TYPE_MAP.get(ann, {"type": "string"})
        props[pname] = json_type
        if param.default is inspect.Parameter.empty:
            required.append(pname)
    schema: dict[str, Any] = {"type": "object", "properties": props}
    if required:
        schema["required"] = required
    return schema


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


def _classify_sdk_error(
    exc: BaseException, agent_label: str
) -> AgenticRunError:
    """Translate a Claude SDK exception into an actionable error.

    Inspects the exception type and (where applicable) its ``stderr``
    field for the most common operator-facing failure modes —
    missing CLI binary, unauthenticated session, missing API key,
    or rate limit — and returns an :class:`AgenticRunError` whose
    message tells the user what to do next.  Falls back to a
    generic translation when the underlying failure does not match
    a known pattern.

    Parameters
    ----------
    exc : BaseException
        The exception raised from inside an SDK call.
    agent_label : str
        Either ``"Strategizer"`` or ``"Implementer"`` so the user
        knows which session failed.

    Returns
    -------
    AgenticRunError
        Ready to be raised by the caller via ``raise ... from exc``.
    """

    try:
        from claude_agent_sdk._errors import (
            CLIConnectionError,
            CLINotFoundError,
            ProcessError,
        )
    except ImportError:
        return AgenticRunError(
            f"{agent_label} SDK call failed: {type(exc).__name__}: "
            f"{exc}. The claude-agent-sdk package is required; "
            'install it with `uv pip install -e ".[agentic]"`.'
        )

    if isinstance(exc, CLINotFoundError):
        return AgenticRunError(
            f"{agent_label} cannot start: the `claude` CLI binary "
            "was not found on PATH. Install Claude Code "
            "(https://docs.claude.com/en/docs/claude-code) and "
            "run `claude` once interactively so it can sign you in "
            "to your Anthropic account."
        )

    if isinstance(exc, ProcessError):
        stderr_text = (exc.stderr or "").lower()
        auth_tokens = (
            "authenticate",
            "not logged in",
            "log in",
            "login required",
            "401",
            "unauthor",
        )
        api_key_tokens = ("api key", "anthropic_api_key", "x-api-key")
        rate_tokens = ("rate limit", "429", "quota", "credit")

        if any(tok in stderr_text for tok in auth_tokens):
            return AgenticRunError(
                f"{agent_label} cannot authenticate with Claude. "
                "Run `claude` once interactively to sign in to your "
                "Anthropic account, then retry the agentic loop. "
                f"\nUnderlying SDK error: {exc}"
            )
        if any(tok in stderr_text for tok in api_key_tokens):
            return AgenticRunError(
                f"{agent_label} reports an API-key problem. From "
                "2026-06-15 onward programmatic Claude Agent SDK "
                "usage may need ANTHROPIC_API_KEY set in your "
                "environment in addition to (or instead of) the "
                "interactive `claude` login. Export the key and "
                f"retry.\nUnderlying SDK error: {exc}"
            )
        if any(tok in stderr_text for tok in rate_tokens):
            return AgenticRunError(
                f"{agent_label} hit a rate or credit limit. Wait "
                "and retry, or check your Claude plan's available "
                f"usage budget.\nUnderlying SDK error: {exc}"
            )
        return AgenticRunError(
            f"{agent_label} CLI subprocess failed "
            f"(exit code {exc.exit_code}). stderr:\n{exc.stderr}"
        )

    if isinstance(exc, CLIConnectionError):
        return AgenticRunError(
            f"{agent_label} could not connect to the Claude CLI: "
            f"{exc}. Check the CLI is installed and authenticated."
        )

    return AgenticRunError(
        f"{agent_label} SDK call failed with an unexpected error: "
        f"{type(exc).__name__}: {exc}"
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
    model : str
        Claude model identifier used by both sessions.
    checkpoint_every : int
        Number of Implementer delegations between checkpoints.
    stdin : TextIO
        Readable stream for user input (defaults to ``sys.stdin``).
    stdout : TextIO
        Writable stream for user-facing output (defaults to
        ``sys.stdout``).
    strategizer_factory : callable or None
        Factory ``(system_prompt, model, tool_closures) ->
        StrategizerSession``.  If ``None``, uses
        ``_ClaudeStrategizer``.
    implementer_factory : callable or None
        Factory ``(system_prompt, model, study_dir) ->
        ImplementerSession``.  If ``None``, uses
        ``_ClaudeImplementer``.
    """

    def __init__(
        self,
        study_dir: Path,
        *,
        model: str = MVP_DEFAULT_MODEL,
        checkpoint_every: int = CHECKPOINT_EVERY,
        stdin: TextIO = sys.stdin,
        stdout: TextIO = sys.stdout,
        strategizer_factory: (
            Callable[..., StrategizerSession] | None
        ) = None,
        implementer_factory: (
            Callable[..., ImplementerSession] | None
        ) = None,
        record_transcripts: bool = True,
    ) -> None:
        self._study_dir = Path(study_dir).resolve()
        self._model = model
        self._checkpoint_every = checkpoint_every
        self._stdin = stdin
        self._stdout = stdout
        self._strategizer_factory = (
            strategizer_factory or _default_strategizer_factory
        )
        self._implementer_factory = (
            implementer_factory or _default_implementer_factory
        )
        self._record_transcripts = record_transcripts

        # Mutable run state — populated in execute().
        self._run_dir: Path | None = None
        self._git_dir: Path | None = None
        self._delegation_counter: int = 0
        self._total_delegations: int = 0
        self._last_report: Report | None = None
        self._done_summary: str | None = None
        self._done_called: bool = False
        self._strategizer: StrategizerSession | None = None
        self._implementer: ImplementerSession | None = None
        self._turn_count: int = 0
        self._checkpoint_summary: str = ""
        self._logger: logging.Logger = logging.getLogger(
            "f3dasm.agentic"
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
        briefing_text = (self._study_dir / "PROBLEM_STATEMENT.md").read_text()
        self._logger.info(
            "Read PROBLEM_STATEMENT.md (%d chars)", len(briefing_text)
        )
        self._start_sessions()
        self._logger.info(
            "Sessions started; handing briefing to Strategizer"
        )
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
        self._preflight_check_claude_cli()

    def _preflight_check_claude_cli(self) -> None:
        """Fail fast if the Claude CLI binary cannot be found.

        The default factories build SDK sessions that delegate to the
        ``claude`` CLI as a subprocess.  When stubs are injected (the
        test path) we skip the check, so unit tests do not need the
        CLI installed.
        """
        using_default_strategizer = (
            self._strategizer_factory is _default_strategizer_factory
        )
        using_default_implementer = (
            self._implementer_factory is _default_implementer_factory
        )
        if not (
            using_default_strategizer or using_default_implementer
        ):
            return
        if shutil.which("claude") is None:
            raise AgenticRunError(
                "Claude CLI not found on PATH. Install Claude Code "
                "(https://docs.claude.com/en/docs/claude-code) and "
                "run `claude` once interactively to sign in to your "
                "Anthropic account before launching the agentic "
                "loop. If the CLI is installed in a non-standard "
                "location, add that directory to PATH."
            )

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
        tool_closures = self._build_strategizer_tools()
        self._strategizer = self._strategizer_factory(
            system_prompt=self._compose_strategizer_prompt(),
            model=self._model,
            tool_closures=tool_closures,
        )
        self._implementer = self._implementer_factory(
            system_prompt=self._compose_implementer_prompt(),
            model=self._model,
            study_dir=self._study_dir,
        )

    def _reset_implementer(self, checkpoint_summary: str) -> None:
        """Destroy the current Implementer and create a fresh one."""
        self._logger.info(
            "Implementer session reset (briefed with checkpoint summary)"
        )
        reset_msg = IMPLEMENTER_RESET_PROMPT_TEMPLATE.format(
            checkpoint_summary=checkpoint_summary
        )
        self._implementer = self._implementer_factory(
            system_prompt=self._compose_implementer_prompt(),
            model=self._model,
            study_dir=self._study_dir,
        )
        self._implementer.send(reset_msg)

    def _compose_strategizer_prompt(self) -> str:
        """Return the Strategizer system prompt with run-specific paths.

        Prepends a small ``<run_paths>`` section so the model knows the
        canonical absolute paths for its notes directory and the study
        tree without having to guess timestamps.
        """
        assert self._run_dir is not None
        notes_dir = (self._run_dir / "strategizer_notes").resolve()
        preamble = RUN_PATHS_PREAMBLE_TEMPLATE.format(
            study_dir=self._study_dir,
            notes_dir=notes_dir,
        )
        return preamble + STRATEGIZER_SYSTEM_PROMPT

    def _compose_implementer_prompt(self) -> str:
        """Return the Implementer system prompt with workspace anchored.

        Prepends a ``<workspace>`` section so the model knows the
        absolute path it must write to. The session's cwd is the study
        directory, so relative paths still resolve correctly; the
        explicit absolute path here is what removes the temptation to
        use ``/tmp``.
        """
        workspace = (self._study_dir / "workspace").resolve()
        preamble = WORKSPACE_PREAMBLE_TEMPLATE.format(
            workspace_dir=workspace,
        )
        return preamble + IMPLEMENTER_SYSTEM_PROMPT

    # ------------------------------------------------------------------
    # Step 5 — Main loop
    # ------------------------------------------------------------------

    def _run_loop(self, briefing_text: str) -> None:
        """Drive the Strategizer until Done() is called or a stop occurs.

        The SDK handles all tool invocations internally (via the in-process
        MCP server). A single ``send()`` call runs the entire multi-turn
        conversation: the model calls tools, gets results, calls more
        tools, and so on until it emits a final message with no tool
        calls. Our ``Done`` and ``Delegate`` closures mutate orchestrator
        state; once ``Done`` is called ``_done_called`` is set and we
        know the run is complete.

        Because the SDK drives the full conversation in one ``send()``,
        the loop here is structurally simple: send the briefing and
        wait. If a ``checkpoint`` resets ``_done_called`` (which it does
        not — checkpoints resume), the second send continues the
        Strategizer session.
        """
        assert self._strategizer is not None
        self._strategizer.send(briefing_text)
        self._turn_count += 1

    # ------------------------------------------------------------------
    # Tool closures (Strategizer)
    # ------------------------------------------------------------------

    def _build_strategizer_tools(
        self,
    ) -> dict[str, Callable[..., str]]:
        """Return the five Strategizer tool closures.

        When ``record_transcripts`` is True each closure is wrapped so
        that every call and result is appended to
        ``runs/<ts>/transcripts/strategizer.jsonl``.
        """
        raw: dict[str, Callable[..., str]] = {
            "Read": self._tool_read,
            "WriteMarkdown": self._tool_write_markdown,
            "Ask": self._tool_ask,
            "Delegate": self._tool_delegate,
            "Done": self._tool_done,
        }
        if not self._record_transcripts:
            return raw
        return {
            name: self._make_transcript_wrapper(name, fn)
            for name, fn in raw.items()
        }

    def _make_transcript_wrapper(
        self,
        tool_name: str,
        fn: Callable[..., str],
    ) -> Callable[..., str]:
        """Return a wrapped version of *fn* that records to strategizer.jsonl.

        Parameters
        ----------
        tool_name : str
            The name used for the ``tool_call`` / ``tool_result`` events.
        fn : callable
            The underlying tool closure.

        Returns
        -------
        callable
            A drop-in replacement that records events before and after
            the underlying call.
        """

        original_sig = inspect.signature(fn)

        def _wrapper(**kwargs: Any) -> str:
            ts_call = (
                datetime.now(tz=timezone.utc)
                .isoformat(timespec="seconds")
            )
            if self._run_dir is not None:
                strat_t = (
                    self._run_dir
                    / "transcripts"
                    / "strategizer.jsonl"
                )
                _record_transcript(
                    strat_t,
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
                    self._run_dir
                    / "transcripts"
                    / "strategizer.jsonl",
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

    def _tool_delegate(self, intent: str, expected_report: str) -> str:
        """Delegate a task to the Implementer.

        Records a git commit, sends the task to the Implementer, parses
        the Report, and returns the raw Report markdown to the
        Strategizer.  If no parseable Report is found, returns an ERROR
        without incrementing the delegation counter.

        Parameters
        ----------
        intent : str
            What the Implementer should do.
        expected_report : str
            What measurements or conclusions are required.

        Returns
        -------
        str
            The Implementer's ``## Report`` block, or an ERROR string.
        """
        assert self._implementer is not None
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
        task = Task(intent=intent, expected_report=expected_report)
        task_msg = _format_task(task)

        # (c) Send to Implementer; record per-delegation transcript.
        impl_reply = self._implementer.send(task_msg)
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
                / f"{deleg_idx}_implementer.jsonl"
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
            retry_reply = self._implementer.send(correction)
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
                strat_transcript = (
                    self._run_dir
                    / "transcripts"
                    / "strategizer.jsonl"
                )
                _record_transcript(
                    strat_transcript,
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
    # Checkpoint
    # ------------------------------------------------------------------

    def _run_checkpoint(self) -> None:
        """Inject the checkpoint prompt and handle user steering."""
        assert self._strategizer is not None

        self._logger.info(
            "Checkpoint firing after %d delegations",
            self._total_delegations,
        )

        self._stdout.write(
            "\n--- CHECKPOINT "
            f"({self._total_delegations} delegations) ---\n"
        )
        self._stdout.flush()

        cp_reply = self._strategizer.send(CHECKPOINT_STRATEGIZER_PROMPT)
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

        # Resume: pass user steering (or "(continue)") to the Strategizer.
        steering_msg = (
            f"user steering: {user_input}"
            if user_input.strip()
            else "(continue)"
        )
        self._strategizer.send(steering_msg)
        self._turn_count += 1

        # Reset the Implementer.
        self._reset_implementer(cp_reply)

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
    return textwrap.dedent(
        f"""\
        ## Task

        ### Intent
        {task.intent}

        ### Expected report
        {task.expected_report}
        """
    )


# ---------------------------------------------------------------------------
# Default factories
# ---------------------------------------------------------------------------


def _default_strategizer_factory(
    *,
    system_prompt: str,
    model: str,
    tool_closures: dict[str, Callable[..., str]],
) -> StrategizerSession:
    """Build a ``_ClaudeStrategizer`` with the supplied closures."""
    return _ClaudeStrategizer(
        system_prompt=system_prompt,
        model=model,
        tool_closures=tool_closures,
    )


def _default_implementer_factory(
    *,
    system_prompt: str,
    model: str,
    study_dir: Path,
) -> ImplementerSession:
    """Build a ``_ClaudeImplementer`` for the given study directory."""
    return _ClaudeImplementer(
        system_prompt=system_prompt,
        model=model,
        study_dir=study_dir,
    )
