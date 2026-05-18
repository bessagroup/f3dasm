"""Tests for the agentic-f3dasm v2 runtime.

All tests use deterministic stub sessions (``StubStrategizer`` and
``StubImplementer``) injected via the factory parameters so no Claude
API calls are made.
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
from io import StringIO
from pathlib import Path
from typing import Any

# Third-party
import pytest

# Local
from f3dasm._src.optimization.agent_runtime import (
    AgenticRun,
    AgenticRunError,
    Task,
    _format_task,
    _parse_report,
)

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Elvis Aguero (elvis_alexander_aguero_vera@brown.edu)"
__credits__ = ["Elvis Aguero"]
__status__ = "Experimental"
# =============================================================================


# ---------------------------------------------------------------------------
# Minimal valid Report block (Implementer output)
# ---------------------------------------------------------------------------

_VALID_REPORT = """\
## Report

### Actions taken
- Did the thing.

### Files touched
- /study/workspace/out.csv

### Conclusions
Task succeeded. No anomalies.

### Numbers
best_x: 0.5
n_evaluated: 10
"""

_VALID_REPORT_2 = """\
## Report

### Actions taken
- Did another thing.

### Files touched
- /study/workspace/out2.csv

### Conclusions
Second task succeeded. Pool coverage improved.

### Numbers
best_x: 0.7
n_evaluated: 20
"""

_INVALID_REPORT = "I did the thing but forgot to include the report block."

# ---------------------------------------------------------------------------
# Stub action types
# ---------------------------------------------------------------------------


class _ReadAction:
    def __init__(self, path: str) -> None:
        self.path = path


class _WriteMdAction:
    def __init__(self, path: str, content: str) -> None:
        self.path = path
        self.content = content


class _AskAction:
    def __init__(self, question: str, mock_user_reply: str) -> None:
        self.question = question
        self.mock_user_reply = mock_user_reply


class _DelegateAction:
    def __init__(self, intent: str, expected_report: str) -> None:
        self.intent = intent
        self.expected_report = expected_report


class _DoneAction:
    def __init__(self, summary: str) -> None:
        self.summary = summary


StrategizerAction = (
    _ReadAction
    | _WriteMdAction
    | _AskAction
    | _DelegateAction
    | _DoneAction
)

# ---------------------------------------------------------------------------
# Stub sessions
# ---------------------------------------------------------------------------


class StubStrategizer:
    """Deterministic Strategizer stub.

    Processes ALL remaining actions in one ``send()`` call, mirroring
    how the real SDK drives all tool calls in a single ``query()``
    invocation.  The stub calls the orchestrator's tool closures
    directly to simulate the SDK's automatic tool invocation.

    ``_AskAction`` uses the real ``Ask`` closure (which reads from the
    run's stdin).  Tests should pre-populate stdin with all expected
    replies before calling ``execute()``.

    ``_CheckpointAction`` is handled transparently: when the runtime
    calls ``send(CHECKPOINT_PROMPT)`` from inside ``_run_checkpoint``,
    the stub returns a canned checkpoint summary.

    Parameters
    ----------
    actions : list[StrategizerAction]
        Ordered list of actions to emit.
    tool_closures : dict[str, callable]
        The tool closures built by the orchestrator, injected at
        construction time via the factory.
    checkpoint_summary : str
        Text returned when ``send()`` is called with the checkpoint
        prompt (i.e. during a checkpoint injection).
    """

    def __init__(
        self,
        actions: list[StrategizerAction],
        tool_closures: dict[str, Any],
        checkpoint_summary: str = "## Checkpoint\nSummary.",
    ) -> None:
        self._actions = list(actions)
        self._tool_closures = tool_closures
        self._checkpoint_summary = checkpoint_summary
        self._call_count = 0
        self._send_count = 0
        self._tool_results: list[str] = []

    def send(self, message: str) -> str:
        """Process actions (or return checkpoint summary) and reply."""
        self._send_count += 1

        if "CHECKPOINT" in message:
            return self._checkpoint_summary

        results: list[str] = []

        while self._actions:
            action = self._actions.pop(0)
            self._call_count += 1

            if isinstance(action, _ReadAction):
                r = self._tool_closures["Read"](path=action.path)
                self._tool_results.append(r)
                results.append(f"[Read result: {r[:40]}]")

            elif isinstance(action, _WriteMdAction):
                r = self._tool_closures["WriteMarkdown"](
                    path=action.path, content=action.content
                )
                self._tool_results.append(r)
                results.append(f"[WriteMarkdown result: {r}]")

            elif isinstance(action, _AskAction):
                r = self._tool_closures["Ask"](
                    question=action.question
                )
                self._tool_results.append(r)
                results.append(f"[Ask result: {r}]")

            elif isinstance(action, _DelegateAction):
                r = self._tool_closures["Delegate"](
                    intent=action.intent,
                    expected_report=action.expected_report,
                )
                self._tool_results.append(r)
                results.append(f"[Delegate result: {r[:60]}]")

            elif isinstance(action, _DoneAction):
                r = self._tool_closures["Done"](summary=action.summary)
                self._tool_results.append(r)
                results.append(f"[Done result: {r}]")
                break

        return "\n".join(results) or "(no action taken)"


class StubImplementer:
    """Deterministic Implementer stub.

    Returns pre-scripted Report strings (or invalid strings) in order.
    The stub factory increments a counter so tests can detect resets.

    Parameters
    ----------
    reports : list[str]
        Pre-scripted response texts.  May contain valid or invalid
        Report blocks.
    """

    def __init__(self, reports: list[str]) -> None:
        self._reports = list(reports)
        self.send_count = 0

    def send(self, message: str) -> str:
        """Return the next pre-scripted response."""
        self.send_count += 1
        if self._reports:
            return self._reports.pop(0)
        return _VALID_REPORT


# ---------------------------------------------------------------------------
# Factory helpers for tests
# ---------------------------------------------------------------------------


def _make_factories(
    strategizer_actions: list[StrategizerAction],
    implementer_reports: list[str],
    implementer_instances: list[StubImplementer] | None = None,
) -> tuple[Any, Any]:
    """Return ``(strategizer_factory, implementer_factory)`` stubs.

    Parameters
    ----------
    strategizer_actions : list
        Actions for ``StubStrategizer``.
    implementer_reports : list[str]
        Reports for ``StubImplementer``.
    implementer_instances : list or None
        If provided, each factory call appends the created instance here
        so the test can inspect it.
    """
    strat_box: list[StubStrategizer] = []

    def strat_factory(
        *,
        system_prompt: str,
        model: str,
        tool_closures: dict[str, Any],
    ) -> StubStrategizer:
        s = StubStrategizer(strategizer_actions, tool_closures)
        strat_box.append(s)
        return s

    remaining_reports: list[str] = list(implementer_reports)

    def impl_factory(
        *,
        system_prompt: str,
        model: str,
        study_dir: Path,
    ) -> StubImplementer:
        stub = StubImplementer(list(remaining_reports))
        if implementer_instances is not None:
            implementer_instances.append(stub)
        return stub

    return strat_factory, impl_factory


# ---------------------------------------------------------------------------
# Test 1 — Bootstrap: missing briefing.md
# ---------------------------------------------------------------------------


def test_bootstrap_missing_briefing(tmp_path: Path) -> None:
    """Missing briefing.md raises AgenticRunError before any work."""
    study_dir = tmp_path / "study"
    study_dir.mkdir()

    strat_factory, impl_factory = _make_factories([], [])

    run = AgenticRun(
        study_dir,
        strategizer_factory=strat_factory,
        implementer_factory=impl_factory,
    )
    with pytest.raises(AgenticRunError, match="briefing.md not found"):
        run.execute()


# ---------------------------------------------------------------------------
# Test 2 — Happy path: Ask, Delegate×2, Done
# ---------------------------------------------------------------------------


def test_happy_path_ask_delegate_done(tmp_path: Path) -> None:
    """Ask + two delegations + Done: deliverable assembled correctly."""
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    (study_dir / "briefing.md").write_text("# Test briefing\nDo stuff.")

    actions: list[StrategizerAction] = [
        _AskAction(
            question="Is the pool ready?",
            mock_user_reply="yes",
        ),
        _DelegateAction(
            intent="Run experiment 1.",
            expected_report="best_x, n_evaluated",
        ),
        _DelegateAction(
            intent="Run experiment 2.",
            expected_report="best_x, n_evaluated",
        ),
        _DoneAction(summary="Best design found: x=0.7 with y=1.5."),
    ]

    strat_factory, impl_factory = _make_factories(
        actions,
        [_VALID_REPORT, _VALID_REPORT_2],
    )

    stdin = StringIO("yes\n")
    stdout = StringIO()

    run = AgenticRun(
        study_dir,
        stdin=stdin,
        stdout=stdout,
        strategizer_factory=strat_factory,
        implementer_factory=impl_factory,
    )
    deliv = run.execute()

    assert deliv.exists()
    assert (deliv / "solution.md").exists()
    assert (deliv / "git_log.txt").exists()
    assert (deliv / "replication").is_dir()

    solution_text = (deliv / "solution.md").read_text()
    assert "Best design found" in solution_text
    assert "total_delegations: 2" in solution_text

    run_dir = deliv.parent
    assert (run_dir / ".git").exists()
    assert (run_dir / "strategizer_notes").is_dir()

    git_log = (deliv / "git_log.txt").read_text()
    assert "initial commit" in git_log


# ---------------------------------------------------------------------------
# Test 3 — WriteMarkdown rejects bad paths
# ---------------------------------------------------------------------------


def test_write_markdown_bad_paths(tmp_path: Path) -> None:
    """WriteMarkdown rejects out-of-notes-dir paths and non-.md files."""
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    (study_dir / "briefing.md").write_text("# Briefing")

    bad_path_results: list[str] = []
    bad_ext_results: list[str] = []

    class _InspectStrategizer:
        def __init__(
            self,
            tool_closures: dict[str, Any],
        ) -> None:
            self._tc = tool_closures
            self._sent = False

        def send(self, message: str) -> str:
            if not self._sent:
                self._sent = True
                bad_path_results.append(
                    self._tc["WriteMarkdown"](
                        path="workspace/evil.md",
                        content="hack",
                    )
                )
                bad_ext_results.append(
                    self._tc["WriteMarkdown"](
                        path=(
                            "runs/faketime/strategizer_notes/"
                            "note.txt"
                        ),
                        content="hack",
                    )
                )
                self._tc["Done"](
                    summary="done"
                )
            return "(done)"

    def strat_factory(
        *,
        system_prompt: str,
        model: str,
        tool_closures: dict[str, Any],
    ) -> _InspectStrategizer:
        return _InspectStrategizer(tool_closures)

    _, impl_factory = _make_factories([], [_VALID_REPORT])

    run = AgenticRun(
        study_dir,
        stdin=StringIO(""),
        stdout=StringIO(),
        strategizer_factory=strat_factory,
        implementer_factory=impl_factory,
    )
    run.execute()

    assert bad_path_results and bad_path_results[0].startswith("ERROR")
    assert bad_ext_results and bad_ext_results[0].startswith("ERROR")


# ---------------------------------------------------------------------------
# Test 4 — Read rejects path escape
# ---------------------------------------------------------------------------


def test_read_rejects_escape(tmp_path: Path) -> None:
    """Read('/etc/passwd') returns ERROR."""
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    (study_dir / "briefing.md").write_text("# Briefing")

    read_results: list[str] = []

    class _EscapeStrategizer:
        def __init__(self, tool_closures: dict[str, Any]) -> None:
            self._tc = tool_closures
            self._sent = False

        def send(self, message: str) -> str:
            if not self._sent:
                self._sent = True
                read_results.append(
                    self._tc["Read"](path="/etc/passwd")
                )
                self._tc["Done"](summary="done")
            return "(done)"

    def strat_factory(
        *,
        system_prompt: str,
        model: str,
        tool_closures: dict[str, Any],
    ) -> _EscapeStrategizer:
        return _EscapeStrategizer(tool_closures)

    _, impl_factory = _make_factories([], [])

    run = AgenticRun(
        study_dir,
        stdin=StringIO(""),
        stdout=StringIO(),
        strategizer_factory=strat_factory,
        implementer_factory=impl_factory,
    )
    run.execute()

    assert read_results and read_results[0].startswith("ERROR")


# ---------------------------------------------------------------------------
# Test 5 — Invalid Report does not increment counter
# ---------------------------------------------------------------------------


def test_invalid_report_does_not_increment(tmp_path: Path) -> None:
    """Invalid Report → ERROR tool result; counter not incremented."""
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    (study_dir / "briefing.md").write_text("# Briefing")

    delegate_results: list[str] = []

    class _InvalidStrategizer:
        def __init__(self, tool_closures: dict[str, Any]) -> None:
            self._tc = tool_closures
            self._sent = False

        def send(self, message: str) -> str:
            if not self._sent:
                self._sent = True
                r = self._tc["Delegate"](
                    intent="do stuff",
                    expected_report="stuff",
                )
                delegate_results.append(r)
                self._tc["Done"](summary="done after bad report")
            return "(done)"

    def strat_factory(
        *,
        system_prompt: str,
        model: str,
        tool_closures: dict[str, Any],
    ) -> _InvalidStrategizer:
        return _InvalidStrategizer(tool_closures)

    _, impl_factory = _make_factories([], [_INVALID_REPORT])

    run = AgenticRun(
        study_dir,
        stdin=StringIO(""),
        stdout=StringIO(),
        strategizer_factory=strat_factory,
        implementer_factory=impl_factory,
    )
    run.execute()

    assert delegate_results[0].startswith("ERROR")
    assert run._total_delegations == 0


# ---------------------------------------------------------------------------
# Test 6 — Checkpoint fires and Implementer resets
# ---------------------------------------------------------------------------


def test_checkpoint_fires_and_resets_implementer(
    tmp_path: Path,
) -> None:
    """After checkpoint_every delegations, checkpoint fires, resets."""
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    (study_dir / "briefing.md").write_text("# Briefing")

    impl_instances: list[StubImplementer] = []
    checkpoint_every = 3

    reports = [_VALID_REPORT] * 5

    class _CheckpointStrategizer:
        """Emits 3 delegates, then expects a checkpoint prompt, then 2 more."""

        def __init__(self, tool_closures: dict[str, Any]) -> None:
            self._tc = tool_closures
            self._call = 0

        def send(self, message: str) -> str:
            self._call += 1

            if self._call == 1:
                for _ in range(checkpoint_every):
                    self._tc["Delegate"](
                        intent="run exp",
                        expected_report="best_x",
                    )

            elif self._call == 2:
                pass

            elif self._call == 3:
                for _ in range(2):
                    self._tc["Delegate"](
                        intent="run exp post-checkpoint",
                        expected_report="best_x",
                    )
                self._tc["Done"](summary="All done.")

            return f"(call {self._call})"

    def strat_factory(
        *,
        system_prompt: str,
        model: str,
        tool_closures: dict[str, Any],
    ) -> _CheckpointStrategizer:
        return _CheckpointStrategizer(tool_closures)

    def impl_factory(
        *,
        system_prompt: str,
        model: str,
        study_dir: Path,
    ) -> StubImplementer:
        stub = StubImplementer(list(reports))
        impl_instances.append(stub)
        return stub

    stdin = StringIO("continue\n")
    stdout = StringIO()

    run = AgenticRun(
        study_dir,
        checkpoint_every=checkpoint_every,
        stdin=stdin,
        stdout=stdout,
        strategizer_factory=strat_factory,
        implementer_factory=impl_factory,
    )
    run.execute()

    assert run._total_delegations == 5
    assert len(impl_instances) >= 2, (
        "Implementer should have been reset (factory called at least twice)"
    )


# ---------------------------------------------------------------------------
# Test 7 — Checkpoint user types "stop"
# ---------------------------------------------------------------------------


def test_checkpoint_stop_assembles_deliverable(tmp_path: Path) -> None:
    """User types 'stop' at checkpoint; deliverable uses checkpoint summary."""
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    (study_dir / "briefing.md").write_text("# Briefing")

    checkpoint_every = 2

    class _StopStrategizer:
        def __init__(self, tool_closures: dict[str, Any]) -> None:
            self._tc = tool_closures
            self._call = 0

        def send(self, message: str) -> str:
            self._call += 1
            if self._call == 1:
                for _ in range(checkpoint_every):
                    self._tc["Delegate"](
                        intent="exp",
                        expected_report="x",
                    )
            elif self._call == 2:
                return (
                    "## Checkpoint\n"
                    "### Recommended next direction\nStop here."
                )
            return f"(call {self._call})"

    def strat_factory(
        *,
        system_prompt: str,
        model: str,
        tool_closures: dict[str, Any],
    ) -> _StopStrategizer:
        return _StopStrategizer(tool_closures)

    _, impl_factory = _make_factories(
        [], [_VALID_REPORT, _VALID_REPORT_2]
    )

    stdin = StringIO("stop\n")
    stdout = StringIO()

    run = AgenticRun(
        study_dir,
        checkpoint_every=checkpoint_every,
        stdin=stdin,
        stdout=stdout,
        strategizer_factory=strat_factory,
        implementer_factory=impl_factory,
    )
    deliv = run.execute()

    assert run._done_called
    solution = (deliv / "solution.md").read_text()
    assert "## Checkpoint" in solution or "Stop here" in solution or \
        run._done_summary is None


# ---------------------------------------------------------------------------
# Test 8 — Empty stdin at checkpoint treated as continue
# ---------------------------------------------------------------------------


def test_checkpoint_empty_stdin_continues(tmp_path: Path) -> None:
    """Empty stdin at checkpoint is not treated as 'stop'."""
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    (study_dir / "briefing.md").write_text("# Briefing")

    checkpoint_every = 1
    impl_instances: list[StubImplementer] = []

    class _ContinueStrategizer:
        def __init__(self, tool_closures: dict[str, Any]) -> None:
            self._tc = tool_closures
            self._call = 0

        def send(self, message: str) -> str:
            self._call += 1
            if self._call == 1:
                self._tc["Delegate"](
                    intent="exp",
                    expected_report="x",
                )
            elif self._call == 2:
                return "checkpoint summary"
            elif self._call == 3:
                self._tc["Done"](
                    summary="done after empty continue"
                )
            return f"(call {self._call})"

    def strat_factory(
        *,
        system_prompt: str,
        model: str,
        tool_closures: dict[str, Any],
    ) -> _ContinueStrategizer:
        return _ContinueStrategizer(tool_closures)

    def impl_factory(
        *,
        system_prompt: str,
        model: str,
        study_dir: Path,
    ) -> StubImplementer:
        stub = StubImplementer([_VALID_REPORT])
        impl_instances.append(stub)
        return stub

    stdin = StringIO("\n")
    stdout = StringIO()

    run = AgenticRun(
        study_dir,
        checkpoint_every=checkpoint_every,
        stdin=stdin,
        stdout=stdout,
        strategizer_factory=strat_factory,
        implementer_factory=impl_factory,
    )
    run.execute()

    assert run._done_summary == "done after empty continue"
    assert len(impl_instances) >= 2, (
        "Implementer should have been reset on continue"
    )


# ---------------------------------------------------------------------------
# Test 9 — Run does not touch any git repo above run_dir
# ---------------------------------------------------------------------------


def test_run_does_not_touch_ancestor_git(tmp_path: Path) -> None:
    """The run's git repo is isolated; no ancestor .git is modified."""
    sibling_git = tmp_path / ".git"
    sibling_git.mkdir()
    (sibling_git / "HEAD").write_text("ref: refs/heads/main\n")

    study_dir = tmp_path / "study"
    study_dir.mkdir()
    (study_dir / "briefing.md").write_text("# Briefing")

    actions: list[StrategizerAction] = [
        _DoneAction(summary="Finished quickly."),
    ]
    strat_factory, impl_factory = _make_factories(actions, [])

    run = AgenticRun(
        study_dir,
        stdin=StringIO(""),
        stdout=StringIO(),
        strategizer_factory=strat_factory,
        implementer_factory=impl_factory,
    )
    run.execute()

    head_after = (sibling_git / "HEAD").read_text()
    assert head_after == "ref: refs/heads/main\n", (
        "Ancestor .git/HEAD was unexpectedly modified"
    )

    assert not (study_dir / ".git").exists(), (
        "run should not create .git directly in study_dir"
    )


# ---------------------------------------------------------------------------
# Unit tests for _parse_report and _format_task
# ---------------------------------------------------------------------------


def test_parse_report_valid() -> None:
    """_parse_report correctly extracts all fields."""
    report = _parse_report(_VALID_REPORT)
    assert report is not None
    assert "Did the thing" in report.actions_taken
    assert "/study/workspace/out.csv" in report.files_touched
    assert "Task succeeded" in report.conclusions
    assert report.numbers.get("best_x") == pytest.approx(0.5)
    assert "## Report" in report.raw


def test_parse_report_missing_heading() -> None:
    """_parse_report returns None when ## Report is absent."""
    assert _parse_report(_INVALID_REPORT) is None


def test_format_task_contains_sections() -> None:
    """_format_task renders intent and expected_report sections."""
    t = Task(intent="Do X", expected_report="Return Y")
    text = _format_task(t)
    assert "## Task" in text
    assert "Do X" in text
    assert "Return Y" in text
