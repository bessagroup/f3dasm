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
    from f3dasm._src.agentic.backends.ollama import _preflight_ollama
    from f3dasm._src.agentic.agent_runtime import AgenticRunError

    with patch(
        "f3dasm._src.agentic.backends.ollama.ollama"
    ) as mock_ollama:
        mock_ollama.list.side_effect = Exception("connection refused")
        with pytest.raises(AgenticRunError, match="Ollama server"):
            _preflight_ollama("qwen2.5:0.5b")


def test_preflight_ollama_model_missing():
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
    from f3dasm._src.agentic.backends.ollama import _OllamaAgentSession

    with patch(
        "f3dasm._src.agentic.backends.ollama.ollama"
    ) as mock_ollama:
        mock_ollama.chat.return_value = _make_ollama_response(
            "Hello from Strategizer"
        )
        session = _OllamaAgentSession(
            system_prompt="sys",
            model="qwen2.5:0.5b",
            native_tools=[],
            closure_tools={"Ask": lambda question: "ok"},
        )
        result = session.send("hi")
        assert result == "Hello from Strategizer"


def test_ollama_strategizer_tool_call_ask():
    from f3dasm._src.agentic.backends.ollama import _OllamaAgentSession

    call = MagicMock()
    call.function.name = "Ask"
    call.function.arguments = {"question": "what is x?"}

    with patch(
        "f3dasm._src.agentic.backends.ollama.ollama"
    ) as mock_ollama:
        mock_ollama.chat.side_effect = [
            _make_ollama_response("", tool_calls=[call]),
            _make_ollama_response("Done asking"),
        ]
        ask_results = []
        session = _OllamaAgentSession(
            system_prompt="sys",
            model="qwen2.5:0.5b",
            native_tools=[],
            closure_tools={
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
    from f3dasm._src.agentic.backends.ollama import _OllamaAgentSession

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
        session = _OllamaAgentSession(
            system_prompt="sys",
            model="qwen2.5:0.5b",
            native_tools=["Bash"],
            closure_tools={},
            study_dir=tmp_path,
        )
        result = session.send("do task")
        assert "## Report" in result


def test_ollama_implementer_bash_output_in_history(tmp_path):
    from f3dasm._src.agentic.backends.ollama import _OllamaAgentSession

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
        session = _OllamaAgentSession(
            system_prompt="sys",
            model="qwen2.5:0.5b",
            native_tools=["Bash"],
            closure_tools={},
            study_dir=tmp_path,
        )
        session.send("do task")
        second_call_messages = captured_histories[1]
        all_content = " ".join(
            str(m.get("content", "")) for m in second_call_messages
        )
        assert "captured_output" in all_content
