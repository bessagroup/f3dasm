"""Claude-CLI-backed backend for agentic-f3dasm.

This module owns all Claude-specific session classes, the SDK-error
classifier, the backend preflight check, the async runner helper, the
SDK tool list constants, the schema-inference helper, and the unified
factory function.  The module-level :data:`CLAUDE_BACKEND` constant is
the ready-to-use :class:`~backends.base.Backend` bundle for Claude.

Importing this module does **not** require the ``claude-agent-sdk``
package to be installed; the SDK imports are deferred to the first
:meth:`_ClaudeAgentSession.send` call so that the rest of the runtime
can be imported freely.
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

import asyncio
import inspect as _inspect
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

# AgenticRunError is defined early in agent_runtime (before the
# backends.claude import statement), so this deferred-style partial-module
# import is safe.  We import nothing else from agent_runtime to avoid
# reaching symbols that have not been defined yet.
from ..agent_runtime import AgenticRunError
from .base import Backend

# ---------------------------------------------------------------------------
# Module constant
# ---------------------------------------------------------------------------

MVP_DEFAULT_MODEL: str = "claude-haiku-4-5-20251001"
"""Default Claude model used when no model is specified.

The identical string is also kept in ``agent_runtime`` so that
``from f3dasm._src.agentic.agent_runtime import MVP_DEFAULT_MODEL``
continues to work without a module-level import cycle.
"""

# ---------------------------------------------------------------------------
# SDK tool lists
# ---------------------------------------------------------------------------

_IMPLEMENTER_DENY_LIST: list[str] = [
    "WebSearch",
    "WebFetch",
    "Task",
    "ExitPlanMode",
    "computer",
]
"""Tools that are always disallowed for all agent sessions."""

# ---------------------------------------------------------------------------
# Async runner helper
# ---------------------------------------------------------------------------


def _run_async_safe(coro: Any) -> None:
    """Run a coroutine safely whether or not an event loop is running.

    When called from within an existing asyncio event loop (e.g. from a
    tool handler inside a running ``asyncio.run()`` session), running a
    second ``asyncio.run()`` would raise
    ``RuntimeError: This event loop is already running``.  This helper
    detects that case and runs the coroutine in a fresh event loop on a
    dedicated background thread, joining the thread before returning so
    the call is effectively synchronous from the caller's perspective.

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
# Schema inference helper
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
    sig = _inspect.signature(fn)
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
        if ann is _inspect.Parameter.empty:
            json_type: dict[str, Any] = {"type": "string"}
        else:
            json_type = _TYPE_MAP.get(ann, {"type": "string"})
        props[pname] = json_type
        if param.default is _inspect.Parameter.empty:
            required.append(pname)
    schema: dict[str, Any] = {"type": "object", "properties": props}
    if required:
        schema["required"] = required
    return schema


# ---------------------------------------------------------------------------
# Claude SDK error classifier
# ---------------------------------------------------------------------------


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
        A label identifying the session (e.g. ``"Agent"``) so the user
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


# ---------------------------------------------------------------------------
# Canonical → Claude tool name mapping
# ---------------------------------------------------------------------------
#
# Every canonical name in NATIVE_TOOL_NAMES maps to the identical Claude
# SDK tool name.  This 1:1 correspondence is a deliberate design choice;
# the canonical vocabulary was chosen to match Claude's naming.
#
#   Canonical  →  Claude SDK
#   ─────────────────────────
#   "Bash"     →  "Bash"
#   "Edit"     →  "Edit"
#   "Glob"     →  "Glob"
#   "Grep"     →  "Grep"
#   "MultiEdit"→  "MultiEdit"
#   "Read"     →  "Read"
#   "Write"    →  "Write"
#
# When adding new backends, document the mapping here and in the backend file.


class _ClaudeAgentSession:
    """Unified Claude Agent SDK session.

    Handles both planner-style sessions (with closure tools via an in-process
    MCP server) and executor-style sessions (with native SDK tools).
    A single session may combine both: ``closure_tools`` builds the MCP
    server; ``native_tools`` restricts the SDK's built-in tools.

    Parameters
    ----------
    system_prompt : str
    model : str
    native_tools : list[str]
        Canonical tool names (from NATIVE_TOOL_NAMES).  Maps 1:1 to Claude
        SDK tool names (see mapping above).  Empty list → no native tools.
    closure_tools : dict[str, Callable]
        Python callables exposed as MCP tools.  Empty dict → no MCP server.
    study_dir : Path or None
        Working directory passed as ``cwd`` to the SDK.
    """

    def __init__(
        self,
        system_prompt: str,
        model: str,
        native_tools: list[str],
        closure_tools: dict[str, Callable[..., str]],
        study_dir: Path | None = None,
    ) -> None:
        self._system_prompt = system_prompt
        self._model = model
        self._native_tools = list(native_tools)
        self._closure_tools = closure_tools
        self._study_dir = study_dir
        self._session_id: str | None = None

    def _build_options(self) -> Any:
        """Build ClaudeAgentOptions combining native tools + MCP closures."""
        from claude_agent_sdk import ClaudeAgentOptions

        mcp_servers: dict[str, Any] = {}
        qualified_mcp_tools: list[str] = []

        if self._closure_tools:
            from claude_agent_sdk import SdkMcpTool, create_sdk_mcp_server

            sdk_tools: list[SdkMcpTool] = []
            for tool_name, fn in self._closure_tools.items():
                schema = _infer_schema_from_callable(fn)

                def _make_handler(
                    bound_fn: Callable[..., str],
                ) -> Callable[[dict[str, Any]], Any]:
                    async def _handler(args: dict[str, Any]) -> dict[str, Any]:
                        try:
                            result = bound_fn(**args)
                        except Exception as exc:
                            return {
                                "content": [
                                    {
                                        "type": "text",
                                        "text": (
                                            f"ERROR: {type(exc).__name__}:"
                                            f" {exc}"
                                        ),
                                    }
                                ],
                                "is_error": True,
                            }
                        text = str(result) if result is not None else ""
                        return {"content": [{"type": "text", "text": text}]}

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

            server_name = "f3dasm_agent_tools"
            mcp_cfg = create_sdk_mcp_server(
                name=server_name,
                tools=sdk_tools if sdk_tools else None,
            )
            mcp_servers[server_name] = mcp_cfg
            qualified_mcp_tools = [
                f"mcp__{server_name}__{t.name}" for t in sdk_tools
            ]

        # Combine: MCP tool names + native SDK tool names
        all_allowed = qualified_mcp_tools + self._native_tools

        return ClaudeAgentOptions(
            system_prompt=self._system_prompt,
            model=self._model,
            cwd=str(self._study_dir) if self._study_dir is not None else None,
            tools=self._native_tools if self._native_tools else [],
            mcp_servers=mcp_servers if mcp_servers else {},
            allowed_tools=all_allowed if all_allowed else [],
            disallowed_tools=_IMPLEMENTER_DENY_LIST,
            permission_mode="bypassPermissions",
            resume=self._session_id,
            strict_mcp_config=True if mcp_servers else False,
        )

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
                raise _classify_sdk_error(exc, "Agent") from exc
            raise

        if result_msg is not None:
            self._session_id = result_msg.session_id

        if result_msg is not None and result_msg.is_error:
            raise AgenticRunError(f"Agent SDK error: {result_msg}")
        return assistant_text


# ---------------------------------------------------------------------------
# Preflight check
# ---------------------------------------------------------------------------


def _preflight_claude_cli() -> None:
    """Raise AgenticRunError if the Claude CLI binary is not on PATH.

    The default factory builds SDK sessions that delegate to the
    ``claude`` CLI as a subprocess.  This function is called during
    :meth:`AgenticRun._preflight` when the Claude backend is active.

    Raises
    ------
    AgenticRunError
        When ``shutil.which("claude")`` returns ``None``.
    """
    if shutil.which("claude") is None:
        raise AgenticRunError(
            "Claude CLI not found on PATH. Install Claude Code "
            "(https://docs.claude.com/en/docs/claude-code) and "
            "run `claude` once interactively to sign in to your "
            "Anthropic account before launching the agentic "
            "loop. If the CLI is installed in a non-standard "
            "location, add that directory to PATH."
        )


# ---------------------------------------------------------------------------
# Unified factory function
# ---------------------------------------------------------------------------


def _session_factory(
    *,
    system_prompt: str,
    model: str,
    native_tools: list[str],
    closure_tools: dict[str, Callable[..., str]],
    study_dir: Path | None = None,
) -> _ClaudeAgentSession:
    """Build a :class:`_ClaudeAgentSession` for any node type."""
    return _ClaudeAgentSession(
        system_prompt=system_prompt,
        model=model,
        native_tools=native_tools,
        closure_tools=closure_tools,
        study_dir=study_dir,
    )


# ---------------------------------------------------------------------------
# Module-level backend bundle
# ---------------------------------------------------------------------------

CLAUDE_BACKEND: Backend = Backend(
    name="claude",
    default_model=MVP_DEFAULT_MODEL,
    session_factory=_session_factory,
    preflight=_preflight_claude_cli,
)
"""Ready-to-use Backend bundle for the Claude CLI backend."""
