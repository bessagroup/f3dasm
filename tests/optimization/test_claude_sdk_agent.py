"""Tests for ClaudeSDKAgent.send() wiring through the Claude Agent SDK.

All tests mock the SDK at the module level so no real network calls are made.
"""
#                                                                       Modules
# =============================================================================
from __future__ import annotations

# Standard
from typing import Any
from unittest.mock import MagicMock

# Third-party
import pytest

# Local
from f3dasm._src.optimization.agent_optimizer import ClaudeSDKAgent

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Elvis Aguero (elvis_alexander_aguero_vera@brown.edu)"
__credits__ = ["Elvis Aguero"]
__status__ = "Experimental"
# =============================================================================

# ---------------------------------------------------------------------------
# Fake SDK message types used in mocks
# ---------------------------------------------------------------------------


class _FakeTextBlock:
    """Minimal stand-in for claude_agent_sdk.TextBlock."""

    def __init__(self, text: str) -> None:
        self.text = text


class _FakeAssistantMessage:
    """Minimal stand-in for claude_agent_sdk.AssistantMessage."""

    def __init__(self, text: str) -> None:
        self.content = [_FakeTextBlock(text)]


class _FakeResultMessage:
    """Minimal stand-in for claude_agent_sdk.ResultMessage."""

    def __init__(self, session_id: str, is_error: bool = False) -> None:
        self.session_id = session_id
        self.is_error = is_error


# ---------------------------------------------------------------------------
# Helper: build a fake async generator that yields an AssistantMessage then
# a ResultMessage, simulating a successful SDK query.
# ---------------------------------------------------------------------------


def _make_fake_query(
    assistant_text: str,
    session_id: str = "sess-abc123",
    is_error: bool = False,
) -> Any:
    """Return a coroutine that yields an assistant then a result message."""

    async def _fake_query(
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        yield _FakeAssistantMessage(assistant_text)
        yield _FakeResultMessage(session_id=session_id, is_error=is_error)

    return _fake_query


# ---------------------------------------------------------------------------
# Fixture: patch the entire claude_agent_sdk namespace inside agent_optimizer
# so no subprocess is spawned.
# ---------------------------------------------------------------------------


@pytest.fixture()
def patched_sdk(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Patch claude_agent_sdk imports inside agent_optimizer."""

    # Create mock objects for each SDK symbol used in send().
    mock_sdk_mcp_tool_cls = MagicMock(name="SdkMcpTool")
    mock_create_server = MagicMock(name="create_sdk_mcp_server")
    mock_options_cls = MagicMock(name="ClaudeAgentOptions")
    fake_assistant_text = "## Rationale\nThis is the agent rationale."
    fake_query = _make_fake_query(fake_assistant_text)

    # Patch the 'from claude_agent_sdk import ...' block by intercepting
    # the module-level import at the point send() executes it.  Patching
    # sys.modules ensures the lazy import inside send() resolves to our
    # fake module without spawning any subprocess.

    fake_module = MagicMock(name="claude_agent_sdk")
    fake_module.AssistantMessage = _FakeAssistantMessage
    fake_module.ResultMessage = _FakeResultMessage
    fake_module.TextBlock = _FakeTextBlock
    fake_module.SdkMcpTool = mock_sdk_mcp_tool_cls
    fake_module.ClaudeAgentOptions = mock_options_cls
    fake_module.create_sdk_mcp_server = mock_create_server
    fake_module.query = fake_query

    import sys

    monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake_module)

    return {
        "module": fake_module,
        "SdkMcpTool": mock_sdk_mcp_tool_cls,
        "create_sdk_mcp_server": mock_create_server,
        "ClaudeAgentOptions": mock_options_cls,
        "assistant_text": fake_assistant_text,
    }


# ---------------------------------------------------------------------------
# Test 1: basic wiring — system prompt, model, and return value
# ---------------------------------------------------------------------------


def test_send_returns_assistant_text(patched_sdk: dict[str, Any]) -> None:
    """send() returns the assistant text from the last AssistantMessage."""

    agent = ClaudeSDKAgent(
        name="strategizer",
        system_prompt="You are a test strategizer.",
        model="claude-haiku-4-5-20251001",
    )

    # Ensure the SDK is marked as available (bypassing the real import).
    agent._sdk_available = True

    result = agent.send("Turn 0. Do something.", tools=[])

    # The return value must be the fake assistant text.
    assert result == patched_sdk["assistant_text"]


# ---------------------------------------------------------------------------
# Test 2: tools are registered and system prompt / model are forwarded
# ---------------------------------------------------------------------------


def test_send_forwards_system_prompt_and_model(
    patched_sdk: dict[str, Any],
) -> None:
    """send() forwards system_prompt and model to ClaudeAgentOptions."""

    system_prompt = "## Title\nTest Agent\n## Role\nDo nothing."
    model = "claude-sonnet-4-5"

    agent = ClaudeSDKAgent(
        name="implementer",
        system_prompt=system_prompt,
        model=model,
    )
    agent._sdk_available = True

    def _my_tool(n_steps: int, params: dict) -> str:
        """A stub tool."""
        return "ok"

    agent.send("Turn 1. Execute strategy.", tools=[_my_tool])

    mock_options_cls = patched_sdk["ClaudeAgentOptions"]
    mock_sdk_mcp_tool = patched_sdk["SdkMcpTool"]

    # ClaudeAgentOptions must have been called at least once.
    assert mock_options_cls.called, "ClaudeAgentOptions was never instantiated"

    call_kwargs = mock_options_cls.call_args.kwargs

    # system_prompt must match exactly.
    assert call_kwargs.get("system_prompt") == system_prompt, (
        f"system_prompt mismatch: {call_kwargs.get('system_prompt')!r}"
    )

    # model must match exactly.
    assert call_kwargs.get("model") == model, (
        f"model mismatch: {call_kwargs.get('model')!r}"
    )

    # SdkMcpTool must have been constructed with name="_my_tool" for the
    # tool callable we passed to send().
    tool_names_used = [
        call.kwargs.get("name") or (call.args[0] if call.args else None)
        for call in mock_sdk_mcp_tool.call_args_list
    ]
    assert "_my_tool" in tool_names_used, (
        f"Expected SdkMcpTool constructed with name='_my_tool', "
        f"got: {tool_names_used}"
    )


# ---------------------------------------------------------------------------
# Test 3: session_id is persisted for conversation resume
# ---------------------------------------------------------------------------


def test_send_persists_session_id(patched_sdk: dict[str, Any]) -> None:
    """send() stores session_id from ResultMessage for the next call."""

    agent = ClaudeSDKAgent(name="strategizer", system_prompt="sp")
    agent._sdk_available = True

    assert agent._session_id is None

    agent.send("First turn.", tools=[])

    # After the first call, _session_id must be the fake session ID.
    assert agent._session_id == "sess-abc123"


# ---------------------------------------------------------------------------
# Test 4: deny_list is forwarded to ClaudeAgentOptions
# ---------------------------------------------------------------------------


def test_send_forwards_deny_list(patched_sdk: dict[str, Any]) -> None:
    """send() passes deny_list to ClaudeAgentOptions.disallowed_tools."""

    deny = ["Bash", "Write"]
    agent = ClaudeSDKAgent(
        name="implementer",
        system_prompt="sp",
        deny_list=deny,
    )
    agent._sdk_available = True

    agent.send("Turn.", tools=[])

    mock_options_cls = patched_sdk["ClaudeAgentOptions"]
    call_kwargs = mock_options_cls.call_args.kwargs

    assert call_kwargs.get("disallowed_tools") == deny, (
        f"disallowed_tools mismatch: {call_kwargs.get('disallowed_tools')!r}"
    )
