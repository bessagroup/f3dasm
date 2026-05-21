"""Tests for the agentic-f3dasm v2 runtime.

All tests use deterministic stub sessions (``StubStrategizer`` and
``StubImplementer``) injected via the factory parameters so no Claude
API calls are made.
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

# Standard
import textwrap
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Any

# Third-party
import pytest

# Local
from f3dasm._src.agentic.agent_runtime import (
    AgenticRun,
    AgenticRunError,
    Delegation,
    Report,
    Task,
    _classify_failed_implementer_response,
    _format_task,
    _parse_report,
    read_transcript,
)
from f3dasm._src.agentic.backends.base import Agent, Edge, Graph
from f3dasm._src.agentic.backends.claude import _classify_sdk_error

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
    def __init__(
        self,
        intent: str,
        expected_report: str,
        target: str = "implementer",
    ) -> None:
        self.intent = intent
        self.expected_report = expected_report
        self.target = target


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
                    target=action.target,
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


def _make_default_graph() -> Graph:
    """Return a minimal 2-node strategizer→implementer Graph for tests."""
    class _DefaultStrat(Agent):
        reset_on_checkpoint = False

    class _DefaultImpl(Agent):
        pass

    return Graph(
        nodes={"strategizer": _DefaultStrat(), "implementer": _DefaultImpl()},
        edges=(Edge("strategizer", "implementer"),),
        entry="strategizer",
    )


# ---------------------------------------------------------------------------
# Test 1 — Bootstrap: missing PROBLEM_STATEMENT.md
# ---------------------------------------------------------------------------


def test_bootstrap_missing_briefing(tmp_path: Path) -> None:
    """Missing PROBLEM_STATEMENT.md raises AgenticRunError before any work."""
    study_dir = tmp_path / "study"
    study_dir.mkdir()

    strat_factory, impl_factory = _make_factories([], [])

    run = AgenticRun(
        study_dir,
        graph=_make_default_graph(),
        strategizer_factory=strat_factory,
        implementer_factory=impl_factory,
    )
    with pytest.raises(AgenticRunError, match="PROBLEM_STATEMENT.md not found"):
        run.execute()


# ---------------------------------------------------------------------------
# Test 2 — Happy path: Ask, Delegate×2, Done
# ---------------------------------------------------------------------------


def test_happy_path_ask_delegate_done(tmp_path: Path) -> None:
    """Ask + two delegations + Done: deliverable assembled correctly."""
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    (study_dir / "PROBLEM_STATEMENT.md").write_text("# Test briefing\nDo stuff.")

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
        graph=_make_default_graph(),
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
    """WriteMarkdown rejects path-escape and non-.md files.

    Bare filenames and relative paths inside ``strategizer_notes/`` are
    accepted (and anchored under the canonical notes directory); the
    rejection cases are absolute paths outside the notes directory and
    any file whose extension is not ``.md``.
    """
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    (study_dir / "PROBLEM_STATEMENT.md").write_text("# Briefing")

    bad_path_results: list[str] = []
    bad_ext_results: list[str] = []
    bare_filename_results: list[str] = []

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
                        path="../../etc/evil.md",
                        content="hack",
                    )
                )
                bad_ext_results.append(
                    self._tc["WriteMarkdown"](
                        path="note.txt",
                        content="hack",
                    )
                )
                bare_filename_results.append(
                    self._tc["WriteMarkdown"](
                        path="hypotheses.md",
                        content="ok",
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
        graph=_make_default_graph(),
        stdin=StringIO(""),
        stdout=StringIO(),
        strategizer_factory=strat_factory,
        implementer_factory=impl_factory,
    )
    run.execute()

    assert bad_path_results and bad_path_results[0].startswith("ERROR")
    assert bad_ext_results and bad_ext_results[0].startswith("ERROR")
    assert bare_filename_results and bare_filename_results[0].startswith(
        "OK: wrote "
    )


# ---------------------------------------------------------------------------
# Test 4 — Read rejects path escape
# ---------------------------------------------------------------------------


def test_read_rejects_escape(tmp_path: Path) -> None:
    """Read('/etc/passwd') returns ERROR."""
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    (study_dir / "PROBLEM_STATEMENT.md").write_text("# Briefing")

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
        graph=_make_default_graph(),
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


def test_one_shot_retry_recovers_invalid_report(tmp_path: Path) -> None:
    """First reply malformed → corrective retry → counter increments.

    The runtime sends the Implementer one focused corrective message
    when ``_parse_report`` fails. If the retry yields a valid Report,
    the delegation is counted and the tool result is the recovered
    raw block (not REFLECT).
    """
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    (study_dir / "PROBLEM_STATEMENT.md").write_text("# Briefing")

    delegate_results: list[str] = []

    class _SingleDelegateStrategizer:
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
                self._tc["Done"](summary="done after retry recovered")
            return "(done)"

    def strat_factory(
        *,
        system_prompt: str,
        model: str,
        tool_closures: dict[str, Any],
    ) -> _SingleDelegateStrategizer:
        return _SingleDelegateStrategizer(tool_closures)

    _, impl_factory = _make_factories(
        [], [_INVALID_REPORT, _VALID_REPORT]
    )

    run = AgenticRun(
        study_dir,
        graph=_make_default_graph(),
        stdin=StringIO(""),
        stdout=StringIO(),
        strategizer_factory=strat_factory,
        implementer_factory=impl_factory,
    )
    run.execute()

    first_result = delegate_results[0]
    assert "## Report" in first_result
    assert not first_result.startswith("REFLECT:")
    assert run._total_delegations == 1


def test_two_invalid_replies_fall_through_to_reflect(
    tmp_path: Path,
) -> None:
    """Two malformed replies in a row → REFLECT; counter stays 0."""
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    (study_dir / "PROBLEM_STATEMENT.md").write_text("# Briefing")

    delegate_results: list[str] = []

    class _SingleDelegateStrategizer:
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
                self._tc["Done"](summary="done after two bad replies")
            return "(done)"

    def strat_factory(
        *,
        system_prompt: str,
        model: str,
        tool_closures: dict[str, Any],
    ) -> _SingleDelegateStrategizer:
        return _SingleDelegateStrategizer(tool_closures)

    _, impl_factory = _make_factories(
        [], [_INVALID_REPORT, _INVALID_REPORT]
    )

    run = AgenticRun(
        study_dir,
        graph=_make_default_graph(),
        stdin=StringIO(""),
        stdout=StringIO(),
        strategizer_factory=strat_factory,
        implementer_factory=impl_factory,
    )
    run.execute()

    assert delegate_results[0].startswith("REFLECT:")
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
    (study_dir / "PROBLEM_STATEMENT.md").write_text("# Briefing")

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
        graph=_make_default_graph(),
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
    (study_dir / "PROBLEM_STATEMENT.md").write_text("# Briefing")

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
        graph=_make_default_graph(),
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
    (study_dir / "PROBLEM_STATEMENT.md").write_text("# Briefing")

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
        graph=_make_default_graph(),
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
    (study_dir / "PROBLEM_STATEMENT.md").write_text("# Briefing")

    actions: list[StrategizerAction] = [
        _DoneAction(summary="Finished quickly."),
    ]
    strat_factory, impl_factory = _make_factories(actions, [])

    run = AgenticRun(
        study_dir,
        graph=_make_default_graph(),
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


def test_eval_count_tracked_from_report_numbers(tmp_path: Path) -> None:
    """_total_eval_count accumulates eval_count from report.numbers."""
    (tmp_path / "PROBLEM_STATEMENT.md").write_text("# test\n")

    _REPORT_WITH_EVALS = (
        "## Report\n\n### Actions taken\n- ran\n\n"
        "### Files touched\n\n### Conclusions\nok\n\n"
        "### Numbers\neval_count: 42\n"
    )

    strat_factory, impl_factory = _make_factories(
        [_DelegateAction("do X", "report"), _DoneAction("done")],
        [_REPORT_WITH_EVALS],
    )

    run = AgenticRun(
        tmp_path,
        graph=_make_default_graph(),
        strategizer_factory=strat_factory,
        implementer_factory=impl_factory,
        stdin=StringIO(""),
        stdout=StringIO(),
        record_transcripts=False,
    )
    run.execute()
    assert run._total_eval_count == 42


def test_eval_count_accumulates_across_delegations(tmp_path: Path) -> None:
    """_total_eval_count sums eval_count across multiple delegations."""
    (tmp_path / "PROBLEM_STATEMENT.md").write_text("# test\n")

    _REPORT_20 = (
        "## Report\n\n### Actions taken\n- ran\n\n"
        "### Files touched\n\n### Conclusions\nok\n\n"
        "### Numbers\neval_count: 20\n"
    )

    strat_factory, impl_factory = _make_factories(
        [
            _DelegateAction("phase 1", "report"),
            _DelegateAction("phase 2", "report"),
            _DoneAction("done"),
        ],
        [_REPORT_20, _REPORT_20],
    )

    run = AgenticRun(
        tmp_path,
        graph=_make_default_graph(),
        strategizer_factory=strat_factory,
        implementer_factory=impl_factory,
        stdin=StringIO(""),
        stdout=StringIO(),
        record_transcripts=False,
    )
    run.execute()
    assert run._total_eval_count == 40


def test_eval_budget_warning_injected_into_task(tmp_path: Path) -> None:
    """When eval_count_remaining <= 10% of eval_budget, warning is rendered."""
    task = Task(
        intent="do X",
        expected_report="report",
        eval_count_remaining=40,
        eval_budget=500,
    )
    rendered = _format_task(task)
    # Warning fires at <= 10% = 50 evals remaining; 40 < 50 → warning present
    assert "40" in rendered
    assert "Evaluation" in rendered


def test_eval_budget_no_warning_when_plenty_remaining(tmp_path: Path) -> None:
    """No eval warning when > 10% budget remains."""
    task = Task(
        intent="do X",
        expected_report="report",
        eval_count_remaining=400,
        eval_budget=500,
    )
    rendered = _format_task(task)
    # 400/500 = 80% remaining → no eval warning
    assert "Evaluation budget" not in rendered


# ---------------------------------------------------------------------------
# NEW Test 13 — Piece B: classifier — short response branch
# ---------------------------------------------------------------------------


def test_classify_short_response() -> None:
    """Short response (<100 chars) returns REFLECT: unusually short."""
    result = _classify_failed_implementer_response("Too brief.")
    assert result.startswith("REFLECT:"), (
        f"Expected REFLECT: prefix, got: {result[:60]!r}"
    )
    assert "unusually short" in result, (
        f"Expected 'unusually short' in diagnosis, got: {result[:120]!r}"
    )


# ---------------------------------------------------------------------------
# NEW Test 14 — Piece B: classifier — capability-limit branch
# ---------------------------------------------------------------------------


def test_classify_capability_limit() -> None:
    """Response containing 'I cannot' triggers capability-limit REFLECT."""
    long_response = (
        "I cannot access this file because I do not have the required "
        "permissions to read files outside the study directory.  This "
        "is a limitation of my current tool configuration."
    )
    result = _classify_failed_implementer_response(long_response)
    assert result.startswith("REFLECT:"), (
        f"Expected REFLECT: prefix, got: {result[:60]!r}"
    )
    assert "capability" in result.lower(), (
        f"Expected 'capability' in diagnosis, got: {result[:120]!r}"
    )


# ---------------------------------------------------------------------------
# NEW Test 15 — Piece B: classifier — missing subsections branch
# ---------------------------------------------------------------------------


def test_classify_missing_subsections() -> None:
    """Report heading present but missing subsections triggers REFLECT."""
    partial_report = (
        "## Report\n\n"
        "### Actions taken\n"
        "- Completed step one.\n\n"
        "Some trailing text about what happened.\n"
        * 3
    )
    result = _classify_failed_implementer_response(partial_report)
    assert result.startswith("REFLECT:"), (
        f"Expected REFLECT: prefix, got: {result[:60]!r}"
    )
    assert "subsection" in result.lower(), (
        f"Expected 'subsection' in diagnosis, got: {result[:160]!r}"
    )
    assert "Files touched" in result or "Conclusions" in result or (
        "Numbers" in result
    ), (
        "Expected at least one missing subsection name in REFLECT message"
    )


# ---------------------------------------------------------------------------
# NEW Test 16 — Piece B: classifier — no Report heading branch
# ---------------------------------------------------------------------------


def test_classify_no_report_heading() -> None:
    """200-char response without ## Report triggers missing-block REFLECT."""
    long_no_report = (
        "Here is a long explanation of what I did.  I ran several "
        "steps including reading the CSV file and computing the "
        "statistics, but I did not use the correct output format. "
        "The values I found were interesting."
    )
    assert len(long_no_report) >= 100
    result = _classify_failed_implementer_response(long_no_report)
    assert result.startswith("REFLECT:"), (
        f"Expected REFLECT: prefix, got: {result[:60]!r}"
    )
    assert "## Report" in result or "report block" in result.lower(), (
        "Expected mention of missing Report block in REFLECT message"
    )


# ---------------------------------------------------------------------------
# NEW Test 17 — Piece B: stub-based re-delegation after REFLECT
# ---------------------------------------------------------------------------


def test_reflect_then_redelegation_with_different_intent(
    tmp_path: Path,
) -> None:
    """After a REFLECT on first delegation, a second delegation succeeds.

    The delegation counter must equal 1 (only the successful delegation
    is counted).  The first tool result must start with REFLECT:.
    """
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    (study_dir / "PROBLEM_STATEMENT.md").write_text("# Briefing")

    first_result_box: list[str] = []

    class _ReflectThenRetryStrategizer:
        def __init__(self, tool_closures: dict[str, Any]) -> None:
            self._tc = tool_closures
            self._sent = False

        def send(self, message: str) -> str:
            if not self._sent:
                self._sent = True
                first = self._tc["Delegate"](
                    intent="do the first thing",
                    expected_report="report stuff",
                )
                first_result_box.append(first)
                self._tc["Delegate"](
                    intent="do a DIFFERENT second thing with more context",
                    expected_report="report stuff clearly",
                )
                self._tc["Done"](summary="Done after retry.")
            return "(done)"

    def strat_factory(
        *,
        system_prompt: str,
        model: str,
        tool_closures: dict[str, Any],
    ) -> _ReflectThenRetryStrategizer:
        return _ReflectThenRetryStrategizer(tool_closures)

    _, impl_factory = _make_factories(
        [],
        [_INVALID_REPORT, _INVALID_REPORT, _VALID_REPORT],
    )

    run = AgenticRun(
        study_dir,
        graph=_make_default_graph(),
        stdin=StringIO(""),
        stdout=StringIO(),
        strategizer_factory=strat_factory,
        implementer_factory=impl_factory,
    )
    run.execute()

    assert first_result_box[0].startswith("REFLECT:"), (
        f"First delegation result should start with REFLECT:, "
        f"got {first_result_box[0][:60]!r}"
    )
    assert run._total_delegations == 1, (
        f"Only the successful delegation should be counted; "
        f"got {run._total_delegations}"
    )


# ---------------------------------------------------------------------------
# NEW Test 18 — Piece D: strategizer transcript written when record_transcripts
# ---------------------------------------------------------------------------


def test_transcript_written_when_record_transcripts_true(
    tmp_path: Path,
) -> None:
    """With record_transcripts=True, strategizer.jsonl is created."""
    study_dir = tmp_path / "study"
    study_dir.mkdir()
    (study_dir / "PROBLEM_STATEMENT.md").write_text("# Briefing")

    actions: list[StrategizerAction] = [
        _DelegateAction(
            intent="Run a simple experiment.",
            expected_report="best_x",
        ),
        _DoneAction(summary="Done."),
    ]

    strat_factory, impl_factory = _make_factories(
        actions, [_VALID_REPORT]
    )

    run = AgenticRun(
        study_dir,
        graph=_make_default_graph(),
        stdin=StringIO(""),
        stdout=StringIO(),
        strategizer_factory=strat_factory,
        implementer_factory=impl_factory,
        record_transcripts=True,
    )
    deliv = run.execute()

    run_dir = deliv.parent
    strat_transcript = run_dir / "transcripts" / "strategizer.jsonl"
    assert strat_transcript.exists(), (
        f"Expected strategizer.jsonl at {strat_transcript}"
    )

    events = read_transcript(strat_transcript)
    assert len(events) > 0, "strategizer.jsonl must contain at least one event"

    deliv_transcript = deliv / "transcripts" / "strategizer.jsonl"
    assert deliv_transcript.exists(), (
        "strategizer.jsonl must be copied to deliverable/transcripts/"
    )


# ---------------------------------------------------------------------------
# Tests for SDK exception classification + Claude CLI preflight
# ---------------------------------------------------------------------------


def test_preflight_missing_claude_cli_raises_friendly_error(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """``execute()`` fails fast when the Claude CLI binary is absent.

    Uses the default factories (which target the real SDK) so the
    pre-flight check fires; ``shutil.which`` is monkey-patched to
    simulate a missing binary.
    """

    from f3dasm._src.agentic import agent_runtime as _runtime

    study_dir = tmp_path / "study"
    study_dir.mkdir()
    (study_dir / "PROBLEM_STATEMENT.md").write_text("# Briefing")

    monkeypatch.setattr(_runtime.shutil, "which", lambda *_a, **_k: None)

    run = AgenticRun(study_dir)
    with pytest.raises(AgenticRunError) as excinfo:
        run.execute()
    msg = str(excinfo.value).lower()
    assert "claude cli not found" in msg
    assert "path" in msg


def test_classify_sdk_error_cli_not_found() -> None:
    """A CLINotFoundError becomes an AgenticRunError that names the CLI."""

    from claude_agent_sdk._errors import CLINotFoundError

    exc = CLINotFoundError(cli_path="/usr/local/bin/claude")
    out = _classify_sdk_error(exc, agent_label="Strategizer")
    assert isinstance(out, AgenticRunError)
    text = str(out).lower()
    assert "claude" in text
    assert "path" in text
    assert "strategizer" in text


def test_classify_sdk_error_authentication() -> None:
    """A ProcessError whose stderr mentions 401 yields an auth hint."""

    from claude_agent_sdk._errors import ProcessError

    exc = ProcessError(
        "subprocess failed",
        exit_code=1,
        stderr="401 Unauthorized: please log in.",
    )
    out = _classify_sdk_error(exc, agent_label="Implementer")
    text = str(out).lower()
    assert "authenticate" in text or "sign in" in text
    assert "implementer" in text


def test_classify_sdk_error_api_key() -> None:
    """A ProcessError mentioning ANTHROPIC_API_KEY yields an API-key hint."""

    from claude_agent_sdk._errors import ProcessError

    exc = ProcessError(
        "subprocess failed",
        exit_code=1,
        stderr="missing ANTHROPIC_API_KEY in environment",
    )
    out = _classify_sdk_error(exc, agent_label="Strategizer")
    text = str(out).lower()
    assert (
        "anthropic_api_key" in text
        or "api-key" in text
        or "api key" in text
    )


def test_classify_sdk_error_rate_limit() -> None:
    """A ProcessError mentioning rate limits yields a quota hint."""

    from claude_agent_sdk._errors import ProcessError

    exc = ProcessError(
        "subprocess failed",
        exit_code=1,
        stderr="429 Too Many Requests: rate limit exceeded",
    )
    out = _classify_sdk_error(exc, agent_label="Strategizer")
    text = str(out).lower()
    assert "rate" in text or "credit" in text or "quota" in text


def test_classify_sdk_error_unknown_subclass_fallback() -> None:
    """Unknown ClaudeSDKError subclasses produce a generic fallback."""

    from claude_agent_sdk._errors import ClaudeSDKError

    exc = ClaudeSDKError("something exotic")
    out = _classify_sdk_error(exc, agent_label="Strategizer")
    assert isinstance(out, AgenticRunError)
    assert "ClaudeSDKError" in str(out) or "unexpected" in str(out).lower()


# ---------------------------------------------------------------------------
# Backend bundle: custom Backend drops in via kwarg
# ---------------------------------------------------------------------------


def test_custom_backend_drops_in_via_kwarg(tmp_path: Path) -> None:
    """Passing a custom ``Backend`` swaps both factories + preflight.

    Verifies the three-piece refactor: ``Backend`` bundles the
    strategizer factory, implementer factory, default model, and
    preflight callable. Constructing a stub bundle and passing it via
    ``AgenticRun(backend=...)`` produces a run that uses the stub
    sessions and the stub preflight only.
    """
    from f3dasm.agentic import Backend

    study_dir = tmp_path / "study"
    study_dir.mkdir()
    (study_dir / "PROBLEM_STATEMENT.md").write_text("# Briefing")

    preflight_called: list[bool] = []

    def stub_preflight() -> None:
        preflight_called.append(True)

    strat_factory, impl_factory = _make_factories(
        [_DoneAction(summary="done")],
        [_VALID_REPORT],
    )

    stub_backend = Backend(
        name="stub",
        default_model="stub-model-v1",
        strategizer_factory=strat_factory,
        implementer_factory=impl_factory,
        preflight=stub_preflight,
    )

    run = AgenticRun(
        study_dir,
        graph=_make_default_graph(),
        backend=stub_backend,
        stdin=StringIO(""),
        stdout=StringIO(),
    )
    deliv = run.execute()

    assert deliv.exists()
    assert preflight_called == [True], (
        "Backend.preflight must fire exactly once during execute()"
    )
    solution = (deliv / "solution.md").read_text()
    assert "done" in solution


# ── StudyConfig ──────────────────────────────────────────────────────────────

def test_load_study_config_missing_file(tmp_path):
    """Absent config.yaml → all defaults."""
    from f3dasm._src.agentic.agent_runtime import _load_study_config
    cfg = _load_study_config(tmp_path)
    assert cfg.model is None
    assert cfg.backend == "claude"
    assert cfg.budget is None
    assert cfg.checkpoint_every is None


def test_load_study_config_full(tmp_path):
    """All keys parsed correctly."""
    from f3dasm._src.agentic.agent_runtime import _load_study_config
    (tmp_path / "config.yaml").write_text(textwrap.dedent("""\
        model: llama3.1:8b
        backend: ollama
        budget: "01:30:00"
        checkpoint_every: 10
    """))
    cfg = _load_study_config(tmp_path)
    assert cfg.model == "llama3.1:8b"
    assert cfg.backend == "ollama"
    assert cfg.budget == timedelta(hours=1, minutes=30)
    assert cfg.checkpoint_every == 10


def test_load_study_config_budget_only(tmp_path):
    """Partial config — only budget set."""
    from f3dasm._src.agentic.agent_runtime import _load_study_config
    (tmp_path / "config.yaml").write_text('budget: "00:45:00"\n')
    cfg = _load_study_config(tmp_path)
    assert cfg.budget == timedelta(minutes=45)
    assert cfg.model is None


def test_load_study_config_unknown_key_raises(tmp_path):
    """Unrecognised key → AgenticRunError."""
    from f3dasm._src.agentic.agent_runtime import (
        AgenticRunError,
        _load_study_config,
    )
    (tmp_path / "config.yaml").write_text("unknown_key: value\n")
    with pytest.raises(AgenticRunError, match="unknown_key"):
        _load_study_config(tmp_path)


def test_load_study_config_bad_budget_raises(tmp_path):
    """Malformed budget string → AgenticRunError."""
    from f3dasm._src.agentic.agent_runtime import (
        AgenticRunError,
        _load_study_config,
    )
    (tmp_path / "config.yaml").write_text('budget: "not-a-time"\n')
    with pytest.raises(AgenticRunError, match="budget"):
        _load_study_config(tmp_path)


def test_load_study_config_eval_budget(tmp_path):
    """eval_budget key parsed as int."""
    from f3dasm._src.agentic.agent_runtime import _load_study_config
    (tmp_path / "config.yaml").write_text("eval_budget: 500\n")
    cfg = _load_study_config(tmp_path)
    assert cfg.eval_budget == 500


def test_load_study_config_eval_budget_zero_raises(tmp_path):
    """eval_budget: 0 must raise AgenticRunError."""
    from f3dasm._src.agentic.agent_runtime import AgenticRunError, _load_study_config
    (tmp_path / "config.yaml").write_text("eval_budget: 0\n")
    with pytest.raises(AgenticRunError, match="positive integer"):
        _load_study_config(tmp_path)


def test_load_study_config_eval_budget_negative_raises(tmp_path):
    """eval_budget: -1 must raise AgenticRunError."""
    from f3dasm._src.agentic.agent_runtime import AgenticRunError, _load_study_config
    (tmp_path / "config.yaml").write_text("eval_budget: -1\n")
    with pytest.raises(AgenticRunError, match="positive integer"):
        _load_study_config(tmp_path)


def test_load_study_config_eval_budget_unknown_key_still_raises(tmp_path):
    """Unknown keys still raise even after adding eval_budget."""
    from f3dasm._src.agentic.agent_runtime import AgenticRunError, _load_study_config
    (tmp_path / "config.yaml").write_text("typo_key: 500\n")
    with pytest.raises(AgenticRunError, match="typo_key"):
        _load_study_config(tmp_path)


# ---------------------------------------------------------------------------
# Task 2 tests — StudyConfig wired into AgenticRun
# ---------------------------------------------------------------------------


def test_agentic_run_accepts_study_config(tmp_path):
    """AgenticRun accepts a StudyConfig and uses its checkpoint_every."""
    from f3dasm._src.agentic.agent_runtime import AgenticRun, StudyConfig
    (tmp_path / "PROBLEM_STATEMENT.md").write_text("# Test\n")
    cfg = StudyConfig(checkpoint_every=5)
    strat_factory, impl_factory = _make_factories(
        [_DoneAction("done")], [_VALID_REPORT]
    )
    run = AgenticRun(
        tmp_path,
        graph=_make_default_graph(),
        study_config=cfg,
        strategizer_factory=strat_factory,
        implementer_factory=impl_factory,
        stdin=StringIO(""),
        stdout=StringIO(),
    )
    assert run._checkpoint_every == 5


def test_agentic_run_cli_overrides_config(tmp_path):
    """Explicit CLI model kwarg overrides config.yaml model."""
    from f3dasm._src.agentic.agent_runtime import AgenticRun, StudyConfig
    (tmp_path / "PROBLEM_STATEMENT.md").write_text("# Test\n")
    cfg = StudyConfig(model="model-from-config")
    strat_factory, impl_factory = _make_factories(
        [_DoneAction("done")], [_VALID_REPORT]
    )
    run = AgenticRun(
        tmp_path,
        graph=_make_default_graph(),
        model="model-from-cli",
        study_config=cfg,
        strategizer_factory=strat_factory,
        implementer_factory=impl_factory,
        stdin=StringIO(""),
        stdout=StringIO(),
    )
    assert run._model == "model-from-cli"


# ---------------------------------------------------------------------------
# Task 3 — _remaining() and pre-delegation budget check
# ---------------------------------------------------------------------------


def test_remaining_no_budget(tmp_path):
    """_remaining() returns None when no budget is set."""
    from f3dasm._src.agentic.agent_runtime import AgenticRun, StudyConfig
    (tmp_path / "PROBLEM_STATEMENT.md").write_text("# t\n")
    strat_factory, impl_factory = _make_factories(
        [_DoneAction("done")], [_VALID_REPORT]
    )
    run = AgenticRun(
        tmp_path,
        graph=_make_default_graph(),
        study_config=StudyConfig(),
        strategizer_factory=strat_factory,
        implementer_factory=impl_factory,
        stdin=StringIO(""),
        stdout=StringIO(),
    )
    run._start_time = datetime.now(tz=timezone.utc)
    assert run._remaining() is None


def test_remaining_with_budget(tmp_path):
    """_remaining() returns a positive timedelta just after start."""
    from f3dasm._src.agentic.agent_runtime import AgenticRun, StudyConfig
    (tmp_path / "PROBLEM_STATEMENT.md").write_text("# t\n")
    strat_factory, impl_factory = _make_factories(
        [_DoneAction("done")], [_VALID_REPORT]
    )
    run = AgenticRun(
        tmp_path,
        graph=_make_default_graph(),
        study_config=StudyConfig(budget=timedelta(hours=1)),
        strategizer_factory=strat_factory,
        implementer_factory=impl_factory,
        stdin=StringIO(""),
        stdout=StringIO(),
    )
    run._start_time = datetime.now(tz=timezone.utc)
    rem = run._remaining()
    assert rem is not None
    assert timedelta(minutes=59) < rem <= timedelta(hours=1)


def test_budget_expired_skips_delegation(tmp_path):
    """When budget is exhausted, _tool_delegate triggers clean shutdown."""
    from f3dasm._src.agentic.agent_runtime import AgenticRun, StudyConfig
    (tmp_path / "PROBLEM_STATEMENT.md").write_text("# t\n")
    strat_factory, impl_factory = _make_factories(
        [_DoneAction("done")], [_VALID_REPORT]
    )
    run = AgenticRun(
        tmp_path,
        graph=_make_default_graph(),
        study_config=StudyConfig(budget=timedelta(seconds=1)),
        strategizer_factory=strat_factory,
        implementer_factory=impl_factory,
        stdin=StringIO(""),
        stdout=StringIO(),
    )
    # Simulate an expired budget (start_time was 1 hour ago).
    run._start_time = datetime.now(tz=timezone.utc) - timedelta(hours=1)
    # Need a real implementer stub; create one inline.
    run._implementer = StubImplementer([_VALID_REPORT])
    # Set up minimal run state so _tool_delegate can run.
    run._run_dir = tmp_path / "runs" / "test_run"
    run._run_dir.mkdir(parents=True)
    run._git_dir = run._run_dir / ".git"
    run._git_dir.mkdir()
    result = run._tool_delegate(
        intent="do something", expected_report="report"
    )
    assert run._done_called
    assert "budget" in result.lower()


# ---------------------------------------------------------------------------
# Task 4 — _format_task with remaining_time / budget fields
# ---------------------------------------------------------------------------


def test_format_task_no_budget():
    """No remaining_time → no time line in formatted output."""
    task = Task(intent="do X", expected_report="report Y")
    rendered = _format_task(task)
    assert "Time remaining" not in rendered


def test_format_task_with_budget():
    """remaining_time present → time line appears in formatted output."""
    task = Task(
        intent="do X",
        expected_report="report Y",
        remaining_time=timedelta(hours=1, minutes=12, seconds=47),
    )
    rendered = _format_task(task)
    assert "Time remaining: 01:12:47" in rendered


def test_format_task_budget_warning():
    """remaining_time < 20% of budget → warning line in formatted output."""
    task = Task(
        intent="do X",
        expected_report="report Y",
        remaining_time=timedelta(minutes=5),
        budget=timedelta(hours=1),   # 5/60 ≈ 8% < 20%
    )
    rendered = _format_task(task)
    assert "nearly exhausted" in rendered


def test_solution_md_includes_budget_metadata(tmp_path):
    """solution.md contains budget and time_used when budget is set."""
    from f3dasm._src.agentic.agent_runtime import AgenticRun, StudyConfig
    (tmp_path / "PROBLEM_STATEMENT.md").write_text("# t\n")
    cfg = StudyConfig(budget=timedelta(hours=1))
    strat_factory, impl_factory = _make_factories(
        [_DoneAction("great result")], []
    )
    run = AgenticRun(
        tmp_path,
        graph=_make_default_graph(),
        study_config=cfg,
        strategizer_factory=strat_factory,
        implementer_factory=impl_factory,
        stdin=StringIO(""),
        stdout=StringIO(),
    )
    deliv = run.execute()
    solution = (deliv / "solution.md").read_text()
    assert "budget: 1:00:00" in solution
    assert "time_used:" in solution


# ---------------------------------------------------------------------------
# Named-role routing tests
# ---------------------------------------------------------------------------


def test_named_nodes_two_executors_routing(tmp_path: Path) -> None:
    """Strategizer can delegate to two named executor nodes."""
    (tmp_path / "PROBLEM_STATEMENT.md").write_text("# test\n")

    calls: dict[str, list[str]] = {"alpha": [], "beta": []}

    _ALPHA_REPORT = (
        "## Report\n\n### Actions taken\n- alpha\n\n"
        "### Files touched\n- /workspace/alpha.csv\n\n"
        "### Conclusions\nAlpha succeeded.\n\n### Numbers\nx: 1.0\n"
    )
    _BETA_REPORT = (
        "## Report\n\n### Actions taken\n- beta\n\n"
        "### Files touched\n- /workspace/beta.csv\n\n"
        "### Conclusions\nBeta succeeded.\n\n### Numbers\nx: 2.0\n"
    )

    class _AlphaAgent(Agent):
        system_prompt = "alpha executor"

    class _BetaAgent(Agent):
        system_prompt = "beta executor"

    class _StrategistAgent(Agent):
        reset_on_checkpoint = False

    def impl_factory(*, system_prompt, model, study_dir):
        if "alpha" in system_prompt:
            class AlphaExec:
                def send(self, msg):
                    calls["alpha"].append(msg)
                    return _ALPHA_REPORT
            return AlphaExec()
        else:
            class BetaExec:
                def send(self, msg):
                    calls["beta"].append(msg)
                    return _BETA_REPORT
            return BetaExec()

    graph = Graph(
        nodes={
            "strategizer": _StrategistAgent(),
            "alpha": _AlphaAgent(),
            "beta": _BetaAgent(),
        },
        edges=(Edge("strategizer", "alpha"), Edge("strategizer", "beta")),
        entry="strategizer",
    )

    strat_factory, _ = _make_factories(
        [
            _DelegateAction("task for alpha", "alpha report", target="alpha"),
            _DelegateAction("task for beta", "beta report", target="beta"),
            _DoneAction("both done"),
        ],
        [],
    )

    run = AgenticRun(
        tmp_path,
        graph=graph,
        strategizer_factory=strat_factory,
        implementer_factory=impl_factory,
        stdin=StringIO(""),
        stdout=StringIO(),
        record_transcripts=False,
    )
    run.execute()

    assert len(calls["alpha"]) >= 1, "alpha executor was never called"
    assert len(calls["beta"]) >= 1, "beta executor was never called"

def test_named_roles_unknown_target_returns_error(tmp_path: Path) -> None:
    """Delegate to an unknown target returns an ERROR string (no crash)."""
    (tmp_path / "PROBLEM_STATEMENT.md").write_text("# test\n")

    received: list[str] = []

    class _RecordingStrategizer:
        def __init__(self, tool_closures: dict) -> None:
            self._closures = tool_closures

        def send(self, message: str) -> str:
            result = self._closures["Delegate"](
                intent="do X",
                expected_report="report Y",
                target="nonexistent_agent",
            )
            received.append(result)
            self._closures["Done"](summary="done")
            return "ok"

    def strat_factory(*, system_prompt, model, tool_closures):
        return _RecordingStrategizer(tool_closures)

    def impl_factory(*, system_prompt, model, study_dir):
        return StubImplementer([_VALID_REPORT])

    run = AgenticRun(
        tmp_path,
        graph=_make_default_graph(),
        strategizer_factory=strat_factory,
        implementer_factory=impl_factory,
        stdin=StringIO(""),
        stdout=StringIO(),
        record_transcripts=False,
    )
    run.execute()

    assert received, "Delegate closure was not called"
    assert received[0].startswith("ERROR:"), (
        f"Expected ERROR string, got: {received[0]!r}"
    )


# ---------------------------------------------------------------------------
# TestAgentClassAPI — Task 1: Agent base class + Graph(nodes=)
# ---------------------------------------------------------------------------


class TestAgentClassAPI:
    def test_agent_base_defaults(self):
        a = Agent()
        assert a.system_prompt == "" and a.tools == frozenset() and a.model is None

    def test_agent_subclass_inherits_and_overrides(self):
        class Impl(Agent):
            system_prompt = "..."
            tools = frozenset({"bash"})

        a = Impl(model="haiku")
        assert a.tools == frozenset({"bash"}) and a.model == "haiku"

    def test_graph_nodes_dict(self):
        class S(Agent):
            pass

        class I(Agent):
            pass

        g = Graph(nodes={"s": S(), "i": I()}, edges=(Edge("s", "i"),), entry="s")
        assert g.outgoing("s") == ["i"] and g.outgoing("i") == []

    def test_graph_rejects_unknown_edge_target(self):
        class A(Agent):
            pass

        with pytest.raises(ValueError, match="undeclared"):
            Graph(nodes={"a": A()}, edges=(Edge("a", "missing"),), entry="a")

    def test_graph_rejects_unknown_entry(self):
        class A(Agent):
            pass

        with pytest.raises(ValueError, match="entry"):
            Graph(nodes={"a": A()}, edges=(), entry="missing")


# ---------------------------------------------------------------------------
# TestAgenticRunWithNodes — Graph.nodes wiring (Task 2)
# ---------------------------------------------------------------------------


class TestAgenticRunWithNodes:
    def test_run_entry_node_receives_briefing(self, tmp_path):
        """graph.entry node's session receives the PROBLEM_STATEMENT.md content."""
        (tmp_path / "PROBLEM_STATEMENT.md").write_text("Test briefing content.")

        received = []

        class StubStrat(Agent):
            system_prompt = "stub strat"
            reset_on_checkpoint = False

        class StubImpl(Agent):
            system_prompt = "stub impl"

        def strat_factory(*, system_prompt, model, tool_closures):
            class S:
                def send(self, msg):
                    received.append(msg)
                    return ""
            return S()

        def impl_factory(*, system_prompt, model, study_dir):
            class I:
                def send(self, msg): return ""
            return I()

        g = Graph(
            nodes={"strategizer": StubStrat(), "implementer": StubImpl()},
            edges=(Edge("strategizer", "implementer"),),
            entry="strategizer",
        )
        run = AgenticRun(
            tmp_path,
            graph=g,
            strategizer_factory=strat_factory,
            implementer_factory=impl_factory,
        )
        run.execute()
        assert "Test briefing content." in received[0]

    def test_node_with_outgoing_edge_gets_planner_session(self, tmp_path):
        """Node with outgoing edges gets tool_closures (planner session factory called)."""
        (tmp_path / "PROBLEM_STATEMENT.md").write_text("brief")
        planner_called = []
        executor_called = []

        class StubStrat(Agent):
            system_prompt = "strat"
            reset_on_checkpoint = False

        class StubImpl(Agent):
            system_prompt = "impl"

        def strat_factory(*, system_prompt, model, tool_closures):
            planner_called.append(True)
            class S:
                def send(self, msg): return ""
            return S()

        def impl_factory(*, system_prompt, model, study_dir):
            executor_called.append(True)
            class I:
                def send(self, msg): return ""
            return I()

        g = Graph(
            nodes={"strategizer": StubStrat(), "implementer": StubImpl()},
            edges=(Edge("strategizer", "implementer"),),
            entry="strategizer",
        )
        run = AgenticRun(
            tmp_path, graph=g,
            strategizer_factory=strat_factory,
            implementer_factory=impl_factory,
        )
        run.execute()
        assert planner_called  # strategizer has outgoing edge → planner session
        assert executor_called  # implementer has no outgoing edge → executor session

    def test_node_without_outgoing_edge_gets_executor_session(self, tmp_path):
        """Node without outgoing edges gets study_dir (executor session factory called)."""
        (tmp_path / "PROBLEM_STATEMENT.md").write_text("brief")
        executor_study_dirs = []

        class StubStrat(Agent):
            system_prompt = "strat"
            reset_on_checkpoint = False

        class StubImpl(Agent):
            system_prompt = "impl"

        def strat_factory(*, system_prompt, model, tool_closures):
            class S:
                def send(self, msg): return ""
            return S()

        def impl_factory(*, system_prompt, model, study_dir):
            executor_study_dirs.append(study_dir)
            class I:
                def send(self, msg): return ""
            return I()

        g = Graph(
            nodes={"strategizer": StubStrat(), "implementer": StubImpl()},
            edges=(Edge("strategizer", "implementer"),),
            entry="strategizer",
        )
        run = AgenticRun(
            tmp_path, graph=g,
            strategizer_factory=strat_factory,
            implementer_factory=impl_factory,
        )
        run.execute()
        assert len(executor_study_dirs) == 1
        assert executor_study_dirs[0] == tmp_path.resolve()

    def test_run_uses_agent_model_override(self, tmp_path):
        """Agent(model='haiku') overrides the run-level model for that node."""
        (tmp_path / "PROBLEM_STATEMENT.md").write_text("brief")
        used_models = {}

        class StubStrat(Agent):
            system_prompt = "strat"
            reset_on_checkpoint = False

        class StubImpl(Agent):
            system_prompt = "impl"

        def strat_factory(*, system_prompt, model, tool_closures):
            used_models["strategizer"] = model
            class S:
                def send(self, msg): return ""
            return S()

        def impl_factory(*, system_prompt, model, study_dir):
            used_models["implementer"] = model
            class I:
                def send(self, msg): return ""
            return I()

        g = Graph(
            nodes={
                "strategizer": StubStrat(),
                "implementer": StubImpl(model="haiku"),
            },
            edges=(Edge("strategizer", "implementer"),),
            entry="strategizer",
        )
        run = AgenticRun(
            tmp_path, graph=g, model="sonnet",
            strategizer_factory=strat_factory,
            implementer_factory=impl_factory,
        )
        run.execute()
        assert used_models["strategizer"] == "sonnet"  # uses run-level model
        assert used_models["implementer"] == "haiku"    # uses agent-level override


# =============================================================================
# TestPlannerToolClosures
# =============================================================================

class TestPlannerToolClosures:
    """Tests for Parallel, Debate, Retry tool closures injected into planner sessions."""

    def _make_run(self, tmp_path, strat_factory, impl_factory):
        """Helper to build AgenticRun with a 2-node graph."""
        from f3dasm._src.agentic.backends.base import Agent, Edge, Graph
        from f3dasm._src.agentic.agent_runtime import AgenticRun

        class Strat(Agent):
            system_prompt = "strat"
            reset_on_checkpoint = False

        class Impl(Agent):
            system_prompt = "impl"

        (tmp_path / "PROBLEM_STATEMENT.md").write_text("brief")
        g = Graph(
            nodes={"strategizer": Strat(), "implementer": Impl()},
            edges=(Edge("strategizer", "implementer"),),
            entry="strategizer",
        )
        return AgenticRun(
            tmp_path, graph=g,
            strategizer_factory=strat_factory,
            implementer_factory=impl_factory,
        )

    def test_parallel_tool_calls_all_targets(self, tmp_path):
        """Parallel tool fans out to each target via _tool_delegate."""
        impl_received = []
        tool_closures_ref = {}

        def strat_factory(*, system_prompt, model, tool_closures):
            tool_closures_ref.update(tool_closures)
            class S:
                def send(self, msg): return ""
            return S()

        def impl_factory(*, system_prompt, model, study_dir):
            class I:
                def send(self, msg):
                    impl_received.append(msg)
                    return "## Report\n### Actions taken\ndone\n### Conclusions\nok"
            return I()

        run = self._make_run(tmp_path, strat_factory, impl_factory)
        run.execute()

        assert "Parallel" in tool_closures_ref
        # Call the Parallel tool directly
        result = tool_closures_ref["Parallel"](
            targets=["implementer", "implementer"],
            intent="do something",
            expected_report="tell me what you did",
        )
        assert "implementer" in result.lower()

    def test_debate_tool_alternates_n_rounds(self, tmp_path):
        """Debate tool sends n rounds alternating between target_a and target_b."""
        tool_closures_ref = {}
        call_count = [0]

        def strat_factory(*, system_prompt, model, tool_closures):
            tool_closures_ref.update(tool_closures)
            class S:
                def send(self, msg): return ""
            return S()

        def impl_factory(*, system_prompt, model, study_dir):
            class I:
                def send(self, msg):
                    call_count[0] += 1
                    return "## Report\n### Actions taken\ndone\n### Conclusions\nreplied"
            return I()

        run = self._make_run(tmp_path, strat_factory, impl_factory)
        run.execute()

        assert "Debate" in tool_closures_ref
        result = tool_closures_ref["Debate"](
            target_a="implementer",
            target_b="implementer",
            n=2,
            initial="Opening statement",
        )
        # 2 rounds x 2 agents = 4 _tool_delegate calls, result has rounds
        assert "Round" in result

    def test_retry_tool_returns_on_first_valid_report(self, tmp_path):
        """Retry tool returns immediately when agent returns a valid ## Report."""
        tool_closures_ref = {}

        def strat_factory(*, system_prompt, model, tool_closures):
            tool_closures_ref.update(tool_closures)
            class S:
                def send(self, msg): return ""
            return S()

        def impl_factory(*, system_prompt, model, study_dir):
            class I:
                def send(self, msg):
                    return "## Report\n### Actions taken\ndone\n### Conclusions\nsuccess"
            return I()

        run = self._make_run(tmp_path, strat_factory, impl_factory)
        run.execute()

        assert "Retry" in tool_closures_ref
        result = tool_closures_ref["Retry"](
            target="implementer",
            intent="do it",
            expected_report="tell me",
        )
        assert "## Report" in result

    def test_retry_tool_retries_on_missing_report(self, tmp_path):
        """Retry tool retries when agent does not return a ## Report block."""
        tool_closures_ref = {}
        attempt_count = [0]

        def strat_factory(*, system_prompt, model, tool_closures):
            tool_closures_ref.update(tool_closures)
            class S:
                def send(self, msg): return ""
            return S()

        def impl_factory(*, system_prompt, model, study_dir):
            class I:
                def send(self, msg):
                    attempt_count[0] += 1
                    if attempt_count[0] < 3:
                        return "No report here yet."
                    return "## Report\n### Actions taken\ndone\n### Conclusions\nfinal"
            return I()

        run = self._make_run(tmp_path, strat_factory, impl_factory)
        run.execute()

        assert "Retry" in tool_closures_ref
        result = tool_closures_ref["Retry"](
            target="implementer",
            intent="do it",
            expected_report="tell me",
            max_attempts=5,
        )
        assert "## Report" in result
        assert attempt_count[0] >= 3
