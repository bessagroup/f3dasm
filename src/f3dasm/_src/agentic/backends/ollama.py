"""Ollama backend for agentic-f3dasm.

Provides ``OLLAMA_BACKEND`` — a :class:`~backends.base.Backend` that
drives agent sessions using a locally-running Ollama server instead of
the Claude Agent SDK.

A single unified session class (_OllamaAgentSession) handles all agent
roles.  It combines closure tools (Done, WriteMarkdown, Delegate, etc.)
with an optional native bash tool (when "Bash" is in native_tools).

The ``ollama`` Python package is imported lazily so that the rest of
agentic-f3dasm can be imported without it installed.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from .base import Backend

__author__ = "Elvis Aguero (elvis_alexander_aguero_vera@brown.edu)"
__credits__ = ["Elvis Aguero"]
__status__ = "Experimental"

OLLAMA_DEFAULT_MODEL: str = "qwen2.5:0.5b"

# Lazy module-level reference; populated on first use via _import_ollama().
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
            "Install it with: uv add 'f3dasm[agentic]'"
        ) from exc


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def _preflight_ollama(model: str) -> None:
    """Check that Ollama is running and *model* is available locally."""
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

def _tool_schema(
    name: str,
    description: str,
    properties: dict,
    required: list[str] | None = None,
) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required if required is not None else list(properties),
            },
        },
    }


_STRATEGIZER_TOOLS = [
    _tool_schema("Ask", "Ask the user a clarifying question.",
                 {"question": {"type": "string"}}),
    _tool_schema(
        "Delegate",
        "Delegate a task to a named agent.",
        {
            "intent": {"type": "string", "description": "What the agent should do."},
            "expected_report": {"type": "string", "description": "Required conclusions."},
            "target": {"type": "string", "description": "Name of agent to delegate to (default: implementer)."},
        },
        required=["intent", "expected_report"],
    ),
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

# Lookup dict for schema generation from closure keys (built once at module level).
_TOOL_SCHEMAS_BY_NAME: dict[str, dict] = {
    s["function"]["name"]: s for s in _STRATEGIZER_TOOLS
}


def _make_delegate_schema(outgoing: list[str]) -> dict:
    """Build a Delegate tool schema with an enum of permitted targets."""
    return _tool_schema(
        "Delegate",
        "Delegate a task to a named agent.",
        {
            "intent": {"type": "string", "description": "What the agent should do."},
            "expected_report": {"type": "string", "description": "Required conclusions."},
            "target": {
                "type": "string",
                "enum": outgoing,
                "description": f"Agent to delegate to. One of: {outgoing}",
            },
        },
        required=["intent", "expected_report", "target"],
    )


def _closures_to_schemas(
    closures: dict, outgoing: list[str]
) -> list[dict]:
    """Build an Ollama tool schema list from a closure dict."""
    schemas = []
    for name in closures:
        if name == "Delegate":
            schemas.append(_make_delegate_schema(outgoing))
        elif name in _TOOL_SCHEMAS_BY_NAME:
            schemas.append(_TOOL_SCHEMAS_BY_NAME[name])
    return schemas

_IMPLEMENTER_TOOLS = [
    _tool_schema("bash",
                 "Execute a shell command in the study directory.",
                 {"cmd": {"type": "string"}}),
]

# ---------------------------------------------------------------------------
# Canonical → Ollama tool name mapping
# ---------------------------------------------------------------------------
#
# Ollama's tool-calling interface is model-dependent and less standardised
# than Claude's.  Only one native tool is supported:
#
#   Canonical  →  Ollama tool
#   ──────────────────────────
#   "Bash"     →  bash (function tool, cmd argument, executed via subprocess)
#
# All other NATIVE_TOOL_NAMES ("Read", "Write", "Edit", "Glob", "Grep",
# "MultiEdit") are NOT natively supported by the Ollama backend.
# Agents that need file I/O in the Ollama backend should use "Bash" and
# call standard Unix commands (cat, echo, find, grep) via bash.
#
# Closure tools (Done, WriteMarkdown, ReadNote, Delegate, etc.) are
# implemented identically to Claude: the LLM model calls them by name
# and the runtime intercepts and executes the Python callable.


def _closures_to_schemas_all(closures: dict) -> list[dict]:
    """Build Ollama tool schema list from any closure dict.

    For known tool names (in _TOOL_SCHEMAS_BY_NAME), uses the cached schema.
    For Delegate, builds a schema with string-typed target.
    For unknown names, generates a minimal schema from the closure dict keys.
    """
    schemas = []
    for name, fn in closures.items():
        if name == "Delegate":
            # Generic delegate schema (target not enum-constrained here)
            schemas.append(
                _tool_schema(
                    "Delegate",
                    "Delegate a task to a named agent.",
                    {
                        "intent": {"type": "string"},
                        "expected_report": {"type": "string"},
                        "target": {"type": "string"},
                    },
                    required=["intent", "expected_report", "target"],
                )
            )
        elif name in _TOOL_SCHEMAS_BY_NAME:
            schemas.append(_TOOL_SCHEMAS_BY_NAME[name])
        else:
            # Generic schema for topology tools (Parallel, Debate, Retry, FollowUp, etc.)
            import inspect as _insp
            sig = _insp.signature(fn)
            props = {
                p: {"type": "string"}
                for p in sig.parameters
            }
            req = [
                p for p, param in sig.parameters.items()
                if param.default is _insp.Parameter.empty
            ]
            schemas.append(_tool_schema(name, fn.__doc__ or name, props, req or None))
    return schemas


# ---------------------------------------------------------------------------
# Unified session class
# ---------------------------------------------------------------------------

class _OllamaAgentSession:
    """Unified Ollama-backed agent session.

    Combines closure tools (any name) with optionally a bash native tool
    (when "Bash" is in native_tools).  All tool calls are handled in a
    single send() loop.

    Parameters
    ----------
    system_prompt : str
    model : str
    native_tools : list[str]
        Canonical names.  Only "Bash" is supported; others are ignored.
    closure_tools : dict[str, callable]
        Protocol and topology tool callables.
    study_dir : Path or None
        Working directory for bash execution.
    """

    def __init__(
        self,
        *,
        system_prompt: str,
        model: str,
        native_tools: list[str],
        closure_tools: dict,
        study_dir: "Path | None" = None,
    ) -> None:
        _import_ollama()
        self._model = model
        self._study_dir = Path(study_dir) if study_dir is not None else Path(".")
        self._closure_tools = closure_tools
        # Build Ollama tool schemas
        self._tool_schemas: list[dict] = []
        # Closure tools schemas (from existing _TOOL_SCHEMAS_BY_NAME or dynamic)
        self._tool_schemas.extend(_closures_to_schemas_all(closure_tools))
        # Native bash tool (if declared)
        if "Bash" in native_tools:
            self._tool_schemas.extend(_IMPLEMENTER_TOOLS)
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
        """Send *message* and run the tool loop until a text reply."""
        self._history.append({"role": "user", "content": message})
        while True:
            response = ollama.chat(
                model=self._model,
                messages=self._history,
                tools=self._tool_schemas,
            )
            msg = response.message
            self._history.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                        }
                        for tc in (msg.tool_calls or [])
                    ],
                }
            )
            if not msg.tool_calls:
                return msg.content or ""
            for call in msg.tool_calls:
                name = call.function.name
                args = call.function.arguments or {}
                if name == "bash":
                    result = self._run_bash(args.get("cmd", ""))
                elif name in self._closure_tools:
                    fn = self._closure_tools[name]
                    try:
                        result = fn(**args)
                    except Exception as exc:
                        result = f"ERROR: {type(exc).__name__}: {exc}"
                else:
                    result = f"ERROR: unknown tool {name!r}. Available: {list(self._closure_tools)}"
                self._history.append({"role": "tool", "content": str(result)})


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def _make_ollama_session(
    *,
    system_prompt: str,
    model: str,
    native_tools: list[str],
    closure_tools: dict,
    study_dir: "Path | None" = None,
) -> _OllamaAgentSession:
    return _OllamaAgentSession(
        system_prompt=system_prompt,
        model=model,
        native_tools=native_tools,
        closure_tools=closure_tools,
        study_dir=study_dir,
    )


# ---------------------------------------------------------------------------
# Public backend instance
# ---------------------------------------------------------------------------

OLLAMA_BACKEND: Backend = Backend(
    name="ollama",
    default_model=OLLAMA_DEFAULT_MODEL,
    session_factory=_make_ollama_session,
    preflight=lambda: None,
)
