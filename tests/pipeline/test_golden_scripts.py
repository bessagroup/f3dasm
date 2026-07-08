"""Byte-identity of generated scripts when waves are opted out.

The files under ``golden/`` were captured from the renderer *before*
wave-based array submission existed (commit 495b01a). A pipeline whose
parallel steps set ``max_jobs_per_task=None`` must render scripts that
are byte-for-byte identical to them: opting out of waves is guaranteed
to leave every generated script unchanged.
"""

import pytest

from ._golden import GOLDEN_DIR, render_golden_scripts

pytestmark = pytest.mark.smoke


def test_opted_out_pipeline_renders_golden_scripts():
    scripts = render_golden_scripts({"max_jobs_per_task": None})
    golden_labels = sorted(p.stem for p in GOLDEN_DIR.glob("*.sh"))
    assert sorted(scripts) == golden_labels
    for label, content in scripts.items():
        golden = (GOLDEN_DIR / f"{label}.sh").read_text()
        assert content == golden, (
            f"generated script {label!r} differs from its golden file; "
            "max_jobs_per_task=None must reproduce the pre-wave renderer "
            "byte-for-byte"
        )
