"""Claude-CLI-backed backend for agentic-f3dasm.

This module owns all Claude-specific session classes, the SDK-error
classifier, the backend preflight check, the async runner helper, the
SDK tool list constants, the schema-inference helper, and the two
factory functions.  The module-level :data:`CLAUDE_BACKEND` constant is
the ready-to-use :class:`~backends.base.Backend` bundle for Claude.

Importing this module does **not** require the ``claude-agent-sdk``
package to be installed; the SDK imports are deferred to the first
:meth:`_ClaudeStrategizer.send` / :meth:`_ClaudeImplementer.send`
call so that the rest of the runtime can be imported freely.
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
from .base import Backend, ImplementerSession, StrategizerSession

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
                                        f"ERROR: {type(exc).__name__}:"
                                        f" {exc}"
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
# Preflight check
# ---------------------------------------------------------------------------


def _preflight_claude_cli() -> None:
    """Raise AgenticRunError if the Claude CLI binary is not on PATH.

    The default factories build SDK sessions that delegate to the
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
# Backend-specific factory functions
# ---------------------------------------------------------------------------


def _strategizer_factory(
    *,
    system_prompt: str,
    model: str,
    tool_closures: dict[str, Callable[..., str]],
) -> StrategizerSession:
    """Build a :class:`_ClaudeStrategizer` with the supplied closures."""
    return _ClaudeStrategizer(
        system_prompt=system_prompt,
        model=model,
        tool_closures=tool_closures,
    )


def _implementer_factory(
    *,
    system_prompt: str,
    model: str,
    study_dir: Path,
) -> ImplementerSession:
    """Build a :class:`_ClaudeImplementer` for the given study directory."""
    return _ClaudeImplementer(
        system_prompt=system_prompt,
        model=model,
        study_dir=study_dir,
    )


# ---------------------------------------------------------------------------
# Module-level backend bundle
# ---------------------------------------------------------------------------

CLAUDE_BACKEND: Backend = Backend(
    name="claude",
    default_model=MVP_DEFAULT_MODEL,
    strategizer_factory=_strategizer_factory,
    implementer_factory=_implementer_factory,
    preflight=_preflight_claude_cli,
)
"""Ready-to-use Backend bundle for the Claude CLI backend."""
