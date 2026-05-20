"""Ollama backend for agentic-f3dasm.

Provides ``OLLAMA_BACKEND`` — a :class:`~backends.base.Backend` that
drives Strategizer and Implementer sessions using a locally-running
Ollama server instead of the Claude Agent SDK.

The Implementer's tool surface is a single ``bash`` tool executed via
``subprocess``.  The Strategizer's tools (Ask, Delegate, Done, Read,
WriteMarkdown) are semantic: the runtime intercepts them, not the backend.

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
    """Strategizer session backed by Ollama.

    Implements the StrategizerSession protocol: send(message: str) -> str.
    Tool closures are injected by the runtime — this class does not know
    about the Implementer or any other agent in the topology.
    """

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
    """Implementer session backed by Ollama with a single bash tool.

    Implements the ImplementerSession protocol: send(message: str) -> str.
    """

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
        """Execute *cmd* in the study directory; return stdout+stderr.

        # shell=True uses /bin/sh on Unix; Windows users need WSL.
        """
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
# Factory functions
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
    preflight=lambda: None,  # model-aware preflight called via _preflight_ollama
)
