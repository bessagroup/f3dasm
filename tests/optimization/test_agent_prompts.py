"""Tests for ``f3dasm._src.optimization.agent_prompts``.

Validates that the four prompt constants shipped with the agentic-f3dasm
v2 runtime satisfy structural, content, and integration requirements.
The runtime behaviour itself is covered by ``test_agent_runtime.py``.

Notes
-----
These are *sanity tests* — they enforce the contract between the prompt
module and the runtime, not the scientific quality of the prose.
"""
#                                                                       Modules
# =============================================================================
from __future__ import annotations

import re

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Elvis Aguero (elvis_alexander_aguero_vera@brown.edu)"
__credits__ = ["Elvis Aguero"]
__status__ = "Experimental"
# =============================================================================


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tag_count(text: str, tag: str) -> tuple[int, int]:
    """Return (opening_count, closing_count) for an XML tag pair.

    Parameters
    ----------
    text : str
        The string to search.
    tag : str
        The bare tag name, e.g. ``"role"``.

    Returns
    -------
    tuple[int, int]
        ``(n_opening, n_closing)`` occurrence counts.
    """
    opening = len(re.findall(rf"<{tag}>", text))
    closing = len(re.findall(rf"</{tag}>", text))
    return opening, closing


def _assert_tag_once(text: str, tag: str) -> None:
    """Assert an XML tag pair appears exactly once in *text*.

    An "exact pair" means exactly one ``</tag>`` closing tag exists.
    Opening tags may appear more than once when the tag name is
    referenced in prose (e.g. ``<output_format>.``), so only the
    closing count is used as the uniqueness signal.

    Parameters
    ----------
    text : str
        The string to search.
    tag : str
        The bare tag name.
    """
    _, closing = _tag_count(text, tag)
    assert closing == 1, (
        f"Expected exactly 1 closing </{tag}>; found {closing}"
    )
    # At least one opening must exist to form a valid pair.
    opening, _ = _tag_count(text, tag)
    assert opening >= 1, (
        f"Expected at least 1 opening <{tag}>; found {opening}"
    )


# ---------------------------------------------------------------------------
# Test 1 — constants exist and are non-empty strings
# ---------------------------------------------------------------------------

def test_constants_exist_and_are_non_empty_strings():
    """All four constants are non-empty strings of minimum required length.

    Notes
    -----
    ``IMPLEMENTER_RESET_PROMPT_TEMPLATE`` has a lower threshold (300 chars)
    because it is a short wrapper template, not a full system prompt.
    """
    from f3dasm._src.optimization.agent_prompts import (
        CHECKPOINT_STRATEGIZER_PROMPT,
        IMPLEMENTER_RESET_PROMPT_TEMPLATE,
        IMPLEMENTER_SYSTEM_PROMPT,
        STRATEGIZER_SYSTEM_PROMPT,
    )

    for name, constant in (
        ("STRATEGIZER_SYSTEM_PROMPT", STRATEGIZER_SYSTEM_PROMPT),
        ("IMPLEMENTER_SYSTEM_PROMPT", IMPLEMENTER_SYSTEM_PROMPT),
        ("CHECKPOINT_STRATEGIZER_PROMPT", CHECKPOINT_STRATEGIZER_PROMPT),
    ):
        assert isinstance(constant, str), f"{name} is not a str"
        assert len(constant) >= 500, (
            f"{name} is too short ({len(constant)} chars); expected >= 500"
        )

    assert isinstance(IMPLEMENTER_RESET_PROMPT_TEMPLATE, str), (
        "IMPLEMENTER_RESET_PROMPT_TEMPLATE is not a str"
    )
    assert len(IMPLEMENTER_RESET_PROMPT_TEMPLATE) >= 300, (
        "IMPLEMENTER_RESET_PROMPT_TEMPLATE too short; expected >= 300"
    )


# ---------------------------------------------------------------------------
# Test 2 — STRATEGIZER_SYSTEM_PROMPT XML structure
# ---------------------------------------------------------------------------

def test_strategizer_xml_sections_appear_exactly_once():
    """Required XML-tagged sections each appear exactly once.

    Notes
    -----
    The runtime relies on these sections to give structure to the prompt;
    duplicate tags would confuse any downstream XML parser.
    """
    from f3dasm._src.optimization.agent_prompts import (
        STRATEGIZER_SYSTEM_PROMPT,
    )

    required_tags = [
        "role",
        "deliverable",
        "operating_principles",
        "failure_modes_to_avoid",
        "tool_usage",
        "output_format",
        "examples",
    ]
    for tag in required_tags:
        _assert_tag_once(STRATEGIZER_SYSTEM_PROMPT, tag)


# ---------------------------------------------------------------------------
# Test 3 — STRATEGIZER named failure modes
# ---------------------------------------------------------------------------

def test_strategizer_names_all_failure_modes():
    """All six cognitive-bias failure modes are named in the prompt.

    Notes
    -----
    The names are matched case-insensitively to allow stylistic variation
    (e.g. ``ANCHORING BIAS`` vs ``anchoring``).
    """
    from f3dasm._src.optimization.agent_prompts import (
        STRATEGIZER_SYSTEM_PROMPT,
    )

    lower = STRATEGIZER_SYSTEM_PROMPT.lower()
    required_terms = [
        "anchoring",
        "confirmation",
        "availability",
        "role drift",
        "sycophancy",
        "premature convergence",
    ]
    for term in required_terms:
        assert term in lower, (
            f"STRATEGIZER_SYSTEM_PROMPT does not mention '{term}'"
        )


# ---------------------------------------------------------------------------
# Test 4 — STRATEGIZER mentions all five tools
# ---------------------------------------------------------------------------

def test_strategizer_mentions_all_five_tools():
    """The Strategizer prompt names every tool the agent may call.

    Notes
    -----
    The tool names are checked as literal substrings; the prompt must
    spell them exactly so the model sees the correct function names.
    """
    from f3dasm._src.optimization.agent_prompts import (
        STRATEGIZER_SYSTEM_PROMPT,
    )

    required_tools = ["Read", "WriteMarkdown", "Ask", "Delegate", "Done"]
    for tool in required_tools:
        assert tool in STRATEGIZER_SYSTEM_PROMPT, (
            f"STRATEGIZER_SYSTEM_PROMPT does not mention tool '{tool}'"
        )


# ---------------------------------------------------------------------------
# Test 5 — STRATEGIZER briefing-clarification ritual
# ---------------------------------------------------------------------------

def test_strategizer_briefing_clarification_ritual():
    """Prompt encodes the Ask-before-hypothesis briefing ritual.

    Notes
    -----
    A loose proximity check (within 150 chars) is used rather than an
    exact sentence match, to remain robust to minor prose edits.
    """
    from f3dasm._src.optimization.agent_prompts import (
        STRATEGIZER_SYSTEM_PROMPT,
    )

    lower = STRATEGIZER_SYSTEM_PROMPT.lower()

    # 'Ask' literal tool name must appear.
    assert "ask" in lower, (
        "STRATEGIZER_SYSTEM_PROMPT does not mention 'Ask' tool"
    )

    # 'clarif' root must appear (covers 'clarification', 'clarify', etc.).
    assert "clarif" in lower, (
        "STRATEGIZER_SYSTEM_PROMPT does not contain 'clarif' root"
    )

    # 'before' and 'clarif' must appear within 150 chars of each other,
    # encoding the ritual that clarification precedes hypothesis formation.
    idx_clarif = lower.find("clarif")
    window = lower[max(0, idx_clarif - 150): idx_clarif + 150]
    assert "before" in window, (
        "STRATEGIZER_SYSTEM_PROMPT does not link 'clarif' and 'before' "
        "within a 150-char window — briefing ritual may be missing"
    )


# ---------------------------------------------------------------------------
# Test 6 — IMPLEMENTER_SYSTEM_PROMPT XML structure
# ---------------------------------------------------------------------------

def test_implementer_xml_sections_appear_exactly_once():
    """All required XML-tagged sections appear exactly once.

    Notes
    -----
    Includes the f3dasm-specific ``<f3dasm_primer>`` tag that teaches the
    Implementer how to use the framework.
    """
    from f3dasm._src.optimization.agent_prompts import (
        IMPLEMENTER_SYSTEM_PROMPT,
    )

    required_tags = [
        "role",
        "deliverable",
        "f3dasm_primer",
        "operating_principles",
        "failure_modes_to_avoid",
        "tool_usage",
        "output_format",
        "examples",
    ]
    for tag in required_tags:
        _assert_tag_once(IMPLEMENTER_SYSTEM_PROMPT, tag)


# ---------------------------------------------------------------------------
# Test 7 — IMPLEMENTER f3dasm primer references
# ---------------------------------------------------------------------------

def test_implementer_f3dasm_primer_references_key_classes():
    """The f3dasm primer section names all critical API classes/samplers.

    Notes
    -----
    Checked as literal substrings (case-sensitive) because the model
    must use the exact class names when generating code.
    """
    from f3dasm._src.optimization.agent_prompts import (
        IMPLEMENTER_SYSTEM_PROMPT,
    )

    required_terms = [
        "Domain",
        "ExperimentData",
        "Latin",
        "Sobol",
        "DataGenerator",
        "LookupDataGenerator",
    ]
    for term in required_terms:
        assert term in IMPLEMENTER_SYSTEM_PROMPT, (
            f"IMPLEMENTER_SYSTEM_PROMPT primer missing '{term}'"
        )


# ---------------------------------------------------------------------------
# Test 8 — IMPLEMENTER report format headings
# ---------------------------------------------------------------------------

def test_implementer_report_format_headings():
    """The output-format section contains the exact required headings.

    Notes
    -----
    The runtime greps for ``## Report`` to extract the Implementer's
    response; the other headings are checked here as a structural contract.
    """
    from f3dasm._src.optimization.agent_prompts import (
        IMPLEMENTER_SYSTEM_PROMPT,
    )

    required_headings = [
        "## Report",
        "### Actions taken",
        "### Files touched",
        "### Conclusions",
        "### Numbers",
    ]
    for heading in required_headings:
        assert heading in IMPLEMENTER_SYSTEM_PROMPT, (
            f"IMPLEMENTER_SYSTEM_PROMPT missing heading '{heading}'"
        )


# ---------------------------------------------------------------------------
# Test 9 — IMPLEMENTER refuses hypothesis-verification tasks
# ---------------------------------------------------------------------------

def test_implementer_refuses_hypothesis_verification():
    """The Implementer prompt explicitly instructs refusal of hypothesis
    verification requests.

    Notes
    -----
    Both ``hypothesis`` and ``refuse`` must appear (case-insensitive) so
    the model is clearly instructed to surface scope violations.
    """
    from f3dasm._src.optimization.agent_prompts import (
        IMPLEMENTER_SYSTEM_PROMPT,
    )

    lower = IMPLEMENTER_SYSTEM_PROMPT.lower()
    assert "hypothesis" in lower, (
        "IMPLEMENTER_SYSTEM_PROMPT does not mention 'hypothesis'"
    )
    assert "refuse" in lower, (
        "IMPLEMENTER_SYSTEM_PROMPT does not mention 'refuse'"
    )


# ---------------------------------------------------------------------------
# Test 10 — No supercompressible leakage
# ---------------------------------------------------------------------------

def test_no_supercompressible_leakage():
    """None of the four prompts contain problem-specific vocabulary.

    Notes
    -----
    The prompts must be problem-agnostic.  Any leakage of terms from the
    supercompressible metamaterial study would bias the agent toward that
    domain.
    """
    from f3dasm._src.optimization.agent_prompts import (
        CHECKPOINT_STRATEGIZER_PROMPT,
        IMPLEMENTER_RESET_PROMPT_TEMPLATE,
        IMPLEMENTER_SYSTEM_PROMPT,
        STRATEGIZER_SYSTEM_PROMPT,
    )

    forbidden_terms = [
        "coilable",
        "sigma_crit",
        "ratio_d",
        "ratio_pitch",
        "ratio_top_diameter",
        "bessa",
        "supercompressible",
    ]
    prompts = {
        "STRATEGIZER_SYSTEM_PROMPT": STRATEGIZER_SYSTEM_PROMPT,
        "IMPLEMENTER_SYSTEM_PROMPT": IMPLEMENTER_SYSTEM_PROMPT,
        "CHECKPOINT_STRATEGIZER_PROMPT": CHECKPOINT_STRATEGIZER_PROMPT,
        "IMPLEMENTER_RESET_PROMPT_TEMPLATE": (
            IMPLEMENTER_RESET_PROMPT_TEMPLATE
        ),
    }
    for const_name, text in prompts.items():
        lower = text.lower()
        for term in forbidden_terms:
            assert term not in lower, (
                f"{const_name} leaks domain-specific term '{term}'"
            )


# ---------------------------------------------------------------------------
# Test 11 — CHECKPOINT_STRATEGIZER_PROMPT contract
# ---------------------------------------------------------------------------

def test_checkpoint_prompt_section_headings():
    """Checkpoint prompt contains the exact headings the runtime expects.

    Notes
    -----
    These headings are case-sensitive because the Strategizer model is
    instructed to reproduce them verbatim in its checkpoint report.
    """
    from f3dasm._src.optimization.agent_prompts import (
        CHECKPOINT_STRATEGIZER_PROMPT,
    )

    required_elements = [
        "## Checkpoint",
        "What we have learned",
        "What we have ruled out",
        "Open questions",
        "Recommended next direction",
    ]
    for element in required_elements:
        assert element in CHECKPOINT_STRATEGIZER_PROMPT, (
            f"CHECKPOINT_STRATEGIZER_PROMPT missing element '{element}'"
        )


# ---------------------------------------------------------------------------
# Test 12 — CHECKPOINT forbids new hypotheses
# ---------------------------------------------------------------------------

def test_checkpoint_prompt_forbids_new_hypotheses():
    """Checkpoint prompt explicitly forbids generating new hypotheses.

    Notes
    -----
    Either of two acceptable phrasings is checked case-insensitively.
    """
    from f3dasm._src.optimization.agent_prompts import (
        CHECKPOINT_STRATEGIZER_PROMPT,
    )

    lower = CHECKPOINT_STRATEGIZER_PROMPT.lower()
    acceptable = [
        "do not generate new hypotheses",
        "no new hypotheses",
    ]
    assert any(phrase in lower for phrase in acceptable), (
        "CHECKPOINT_STRATEGIZER_PROMPT does not forbid new hypotheses; "
        f"expected one of: {acceptable}"
    )


# ---------------------------------------------------------------------------
# Test 13 — IMPLEMENTER_RESET_PROMPT_TEMPLATE placeholder
# ---------------------------------------------------------------------------

def test_reset_template_has_exactly_one_placeholder():
    """Template contains ``{checkpoint_summary}`` exactly once and no other
    format placeholders.

    Notes
    -----
    A ``KeyError`` from ``.format()`` would indicate a stray placeholder
    that the runtime does not supply.
    """
    from f3dasm._src.optimization.agent_prompts import (
        IMPLEMENTER_RESET_PROMPT_TEMPLATE,
    )

    assert IMPLEMENTER_RESET_PROMPT_TEMPLATE.count(
        "{checkpoint_summary}"
    ) == 1, (
        "IMPLEMENTER_RESET_PROMPT_TEMPLATE must contain "
        "'{checkpoint_summary}' exactly once"
    )

    # Must not raise KeyError from any unknown placeholder.
    try:
        IMPLEMENTER_RESET_PROMPT_TEMPLATE.format(
            checkpoint_summary="STUB"
        )
    except KeyError as exc:
        raise AssertionError(
            f"IMPLEMENTER_RESET_PROMPT_TEMPLATE has unknown placeholder: "
            f"{exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Test 14 — Reset template structure after formatting
# ---------------------------------------------------------------------------

def test_reset_template_structure_after_formatting():
    """Formatted template contains the stub text plus required anchors.

    Notes
    -----
    ``Strategizer`` and ``Task`` are checked as loose anchors that orient
    the fresh Implementer session to its role and workflow.
    """
    from f3dasm._src.optimization.agent_prompts import (
        IMPLEMENTER_RESET_PROMPT_TEMPLATE,
    )

    result = IMPLEMENTER_RESET_PROMPT_TEMPLATE.format(
        checkpoint_summary="STUB_SUMMARY_TEXT"
    )
    assert "STUB_SUMMARY_TEXT" in result, (
        "Formatted reset template does not contain the substituted summary"
    )
    assert "Strategizer" in result, (
        "Formatted reset template does not reference 'Strategizer'"
    )
    assert "Task" in result, (
        "Formatted reset template does not reference 'Task'"
    )


# ---------------------------------------------------------------------------
# Test 15 — Runtime integration: agent_runtime imports all four constants
# ---------------------------------------------------------------------------

def test_runtime_imports_all_four_constants():
    """``agent_runtime.py`` imports all four prompt constants from
    ``agent_prompts``.

    Notes
    -----
    This is a textual grep check; it does not execute the runtime module
    (which requires the Claude Agent SDK).  The check is intentionally
    lightweight: we verify the source file references the names, not that
    the import succeeds at runtime.
    """
    import pathlib

    runtime_path = pathlib.Path(
        "src/f3dasm/_src/optimization/agent_runtime.py"
    )
    # Resolve relative to the repo root (two parents above tests/).
    if not runtime_path.is_absolute():
        repo_root = (
            pathlib.Path(__file__).parent.parent.parent
        )
        runtime_path = repo_root / runtime_path

    assert runtime_path.exists(), (
        f"agent_runtime.py not found at {runtime_path}"
    )

    source = runtime_path.read_text(encoding="utf-8")

    required_names = [
        "STRATEGIZER_SYSTEM_PROMPT",
        "IMPLEMENTER_SYSTEM_PROMPT",
        "CHECKPOINT_STRATEGIZER_PROMPT",
        "IMPLEMENTER_RESET_PROMPT_TEMPLATE",
    ]
    for name in required_names:
        assert name in source, (
            f"agent_runtime.py does not reference '{name}'"
        )


# ---------------------------------------------------------------------------
# NEW Test 16 — Piece A: hypothesis_log tag appears exactly once
# ---------------------------------------------------------------------------

def test_strategizer_hypothesis_log_tag_appears_once():
    """STRATEGIZER_SYSTEM_PROMPT has exactly one <hypothesis_log> pair."""
    from f3dasm._src.optimization.agent_prompts import (
        STRATEGIZER_SYSTEM_PROMPT,
    )

    _assert_tag_once(STRATEGIZER_SYSTEM_PROMPT, "hypothesis_log")


# ---------------------------------------------------------------------------
# NEW Test 17 — Piece A: hypothesis_log section content
# ---------------------------------------------------------------------------

def test_strategizer_hypothesis_log_content():
    """hypothesis_log section mentions all required fields."""
    from f3dasm._src.optimization.agent_prompts import (
        STRATEGIZER_SYSTEM_PROMPT,
    )

    lower = STRATEGIZER_SYSTEM_PROMPT.lower()
    required_terms = [
        "hypotheses.md",
        "comment",
        "confidence",
        "evidence",
        "status",
        "last_updated_delegation",
    ]
    for term in required_terms:
        assert term in lower, (
            f"STRATEGIZER_SYSTEM_PROMPT hypothesis_log missing '{term}'"
        )


# ---------------------------------------------------------------------------
# NEW Test 18 — Piece B: on_error tag appears exactly once
# ---------------------------------------------------------------------------

def test_strategizer_on_error_tag_appears_once():
    """STRATEGIZER_SYSTEM_PROMPT has exactly one <on_error> pair."""
    from f3dasm._src.optimization.agent_prompts import (
        STRATEGIZER_SYSTEM_PROMPT,
    )

    _assert_tag_once(STRATEGIZER_SYSTEM_PROMPT, "on_error")


# ---------------------------------------------------------------------------
# NEW Test 19 — Piece B: on_error section content
# ---------------------------------------------------------------------------

def test_strategizer_on_error_content():
    """on_error mentions REFLECT: and forbids identical re-delegation."""
    from f3dasm._src.optimization.agent_prompts import (
        STRATEGIZER_SYSTEM_PROMPT,
    )

    assert "REFLECT:" in STRATEGIZER_SYSTEM_PROMPT, (
        "STRATEGIZER_SYSTEM_PROMPT on_error does not mention 'REFLECT:'"
    )
    lower = STRATEGIZER_SYSTEM_PROMPT.lower()
    forbidden_phrases = [
        "forbidden",
        "exact same intent",
    ]
    for phrase in forbidden_phrases:
        assert phrase in lower, (
            f"STRATEGIZER_SYSTEM_PROMPT on_error missing phrase '{phrase}'"
        )


# ---------------------------------------------------------------------------
# NEW Test 20 — Piece A: CHECKPOINT prompt contains Comment log section
# ---------------------------------------------------------------------------

def test_checkpoint_prompt_contains_comment_log():
    """CHECKPOINT_STRATEGIZER_PROMPT contains the ### Comment log section."""
    from f3dasm._src.optimization.agent_prompts import (
        CHECKPOINT_STRATEGIZER_PROMPT,
    )

    assert "### Comment log" in CHECKPOINT_STRATEGIZER_PROMPT, (
        "CHECKPOINT_STRATEGIZER_PROMPT missing '### Comment log' section"
    )


# ---------------------------------------------------------------------------
# NEW Test 21 — Piece C: reasoning_protocol tag appears exactly once
# ---------------------------------------------------------------------------

def test_implementer_reasoning_protocol_tag_appears_once():
    """IMPLEMENTER_SYSTEM_PROMPT has exactly one <reasoning_protocol> pair."""
    from f3dasm._src.optimization.agent_prompts import (
        IMPLEMENTER_SYSTEM_PROMPT,
    )

    _assert_tag_once(IMPLEMENTER_SYSTEM_PROMPT, "reasoning_protocol")


# ---------------------------------------------------------------------------
# NEW Test 22 — Piece C: reasoning_protocol section content
# ---------------------------------------------------------------------------

def test_implementer_reasoning_protocol_content():
    """reasoning_protocol mentions Stages 1-3 and their headings."""
    from f3dasm._src.optimization.agent_prompts import (
        IMPLEMENTER_SYSTEM_PROMPT,
    )

    required_terms = [
        "Stage 1",
        "Stage 2",
        "Stage 3",
        "Task restatement",
        "Workspace inventory",
        "Execution plan",
    ]
    for term in required_terms:
        assert term in IMPLEMENTER_SYSTEM_PROMPT, (
            f"IMPLEMENTER_SYSTEM_PROMPT reasoning_protocol missing "
            f"'{term}'"
        )


# ---------------------------------------------------------------------------
# NEW Test 23 — Piece A/B/C: cleanliness invariant (extended)
# ---------------------------------------------------------------------------

def test_no_domain_specific_leakage_extended():
    """No prompt contains 'coilable', 'sigma_crit', or 'Bessa'."""
    from f3dasm._src.optimization.agent_prompts import (
        CHECKPOINT_STRATEGIZER_PROMPT,
        IMPLEMENTER_RESET_PROMPT_TEMPLATE,
        IMPLEMENTER_SYSTEM_PROMPT,
        STRATEGIZER_SYSTEM_PROMPT,
    )

    forbidden_terms = ["coilable", "sigma_crit", "bessa"]
    prompts = {
        "STRATEGIZER_SYSTEM_PROMPT": STRATEGIZER_SYSTEM_PROMPT,
        "IMPLEMENTER_SYSTEM_PROMPT": IMPLEMENTER_SYSTEM_PROMPT,
        "CHECKPOINT_STRATEGIZER_PROMPT": CHECKPOINT_STRATEGIZER_PROMPT,
        "IMPLEMENTER_RESET_PROMPT_TEMPLATE": (
            IMPLEMENTER_RESET_PROMPT_TEMPLATE
        ),
    }
    for const_name, text in prompts.items():
        lower = text.lower()
        for term in forbidden_terms:
            assert term not in lower, (
                f"{const_name} leaks term '{term}'"
            )
