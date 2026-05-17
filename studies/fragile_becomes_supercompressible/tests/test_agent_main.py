"""Smoke tests for the supercompressible agentic entry point.

These tests verify end-to-end wiring using a 20-row toy pool so that
the full 1000-row pool load is not required for CI speed.  The real
``_StubAgent`` is used throughout — no live Claude SDK calls are made.
"""
#                                                                       Modules
# =============================================================================
from __future__ import annotations

# Standard
import sys
from pathlib import Path

# Third-party
import pytest

# Resolve the study directory so imports work regardless of where pytest
# is invoked from.
_STUDY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_STUDY_DIR))

# Local — study modules under test
from supercompressible_objective import (  # noqa: E402
    SupercompressibleObjective,
    _compute_objective,
)

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Elvis Aguero (elvis_alexander_aguero_vera@brown.edu)"
__credits__ = ["Elvis Aguero"]
__status__ = "Experimental"
# =============================================================================


# =============================================================================
# Helpers — 20-row toy pool
# =============================================================================

# Add repo src to path for f3dasm imports.
_REPO_ROOT = _STUDY_DIR.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from f3dasm import ExperimentData  # noqa: E402
from f3dasm.design import Domain  # noqa: E402


def _make_toy_pool() -> ExperimentData:
    """Return a 20-row in-memory pool covering all three coilable classes.

    Rows 0–9 are coilable=1 (positive sigma_crit), rows 10–14 are
    coilable=0, rows 15–19 are coilable=2.  The grid spans the full
    3D parameter range so the LookupDataGenerator can always find a
    nearest neighbour.
    """
    domain = Domain()
    domain.add_float("ratio_d",            low=0.004, high=0.073)
    domain.add_float("ratio_pitch",        low=0.25,  high=1.50)
    domain.add_float("ratio_top_diameter", low=0.0,   high=0.8)
    domain.add_output("coilable",   to_disk=False, exist_ok=True)
    domain.add_output("sigma_crit", to_disk=False, exist_ok=True)
    domain.add_output("energy",     to_disk=False, exist_ok=True)

    inputs = []
    outputs = []

    # Ten coilable=1 rows with varying sigma_crit.
    for i in range(10):
        ratio_d = 0.004 + i * 0.005
        ratio_pitch = 0.5 + i * 0.05
        ratio_top = 0.05 + i * 0.03
        sigma = 0.1 + i * 0.05
        inputs.append({
            "ratio_d": ratio_d,
            "ratio_pitch": ratio_pitch,
            "ratio_top_diameter": ratio_top,
        })
        outputs.append({"coilable": 1.0, "sigma_crit": sigma, "energy": 0.5})

    # Five coilable=0 rows.
    for i in range(5):
        inputs.append({
            "ratio_d": 0.06 + i * 0.002,
            "ratio_pitch": 1.2 + i * 0.04,
            "ratio_top_diameter": 0.7 + i * 0.02,
        })
        outputs.append({"coilable": 0.0, "sigma_crit": 0.0, "energy": 0.0})

    # Five coilable=2 rows.
    for i in range(5):
        inputs.append({
            "ratio_d": 0.065 + i * 0.001,
            "ratio_pitch": 0.3 + i * 0.02,
            "ratio_top_diameter": 0.6 + i * 0.02,
        })
        outputs.append({"coilable": 2.0, "sigma_crit": 0.0, "energy": 0.0})

    return ExperimentData(
        domain=domain,
        input_data=inputs,
        output_data=outputs,
    )


# =============================================================================
# Tests — SupercompressibleObjective unit tests
# =============================================================================


class TestComputeObjective:
    """Unit tests for the ``_compute_objective`` helper function."""

    def test_coilable_1_returns_sigma_crit(self) -> None:
        """coilable=1 should return sigma_crit unchanged."""
        result = _compute_objective(coilable=1, sigma_crit=0.42, penalty=-1e6)
        assert result == pytest.approx(0.42)

    def test_coilable_0_returns_penalty(self) -> None:
        """coilable=0 (wrong buckling mode) should return the penalty."""
        result = _compute_objective(coilable=0, sigma_crit=0.5, penalty=-1e6)
        assert result == pytest.approx(-1e6)

    def test_coilable_2_returns_penalty(self) -> None:
        """coilable=2 (fractures) should receive the same penalty as 0."""
        result = _compute_objective(coilable=2, sigma_crit=0.3, penalty=-1e6)
        assert result == pytest.approx(-1e6)

    def test_custom_penalty(self) -> None:
        """Custom penalty value is honoured."""
        result = _compute_objective(coilable=0, sigma_crit=1.0, penalty=-999.0)
        assert result == pytest.approx(-999.0)


class TestSupercompressibleObjective:
    """Integration tests for the ``SupercompressibleObjective`` block."""

    def test_objective_column_added(self) -> None:
        """call() adds the 'objective' column to the domain."""
        pool = _make_toy_pool()
        block = SupercompressibleObjective()
        block.call(pool)
        assert "objective" in pool.domain.output_space

    def test_coilable_1_rows_get_positive_objective(self) -> None:
        """All coilable=1 rows receive objective = sigma_crit > 0."""
        pool = _make_toy_pool()
        block = SupercompressibleObjective()
        block.call(pool)
        for sample in pool.data.values():
            coilable = sample._output_data.get("coilable")
            if int(coilable) == 1:
                obj = sample._output_data["objective"]
                sigma = sample._output_data["sigma_crit"]
                assert obj == pytest.approx(sigma), (
                    f"Expected objective={sigma}, got {obj}"
                )

    def test_coilable_0_rows_get_penalty(self) -> None:
        """All coilable=0 rows receive the penalty value."""
        pool = _make_toy_pool()
        block = SupercompressibleObjective(penalty=-1e6)
        block.call(pool)
        for sample in pool.data.values():
            coilable = sample._output_data.get("coilable")
            if int(coilable) == 0:
                obj = sample._output_data["objective"]
                assert obj == pytest.approx(-1e6), (
                    f"coilable=0 row expected penalty -1e6, got {obj}"
                )

    def test_coilable_2_rows_get_penalty(self) -> None:
        """coilable=2 rows receive the same penalty as coilable=0."""
        pool = _make_toy_pool()
        block = SupercompressibleObjective(penalty=-1e6)
        block.call(pool)
        for sample in pool.data.values():
            coilable = sample._output_data.get("coilable")
            if int(coilable) == 2:
                obj = sample._output_data["objective"]
                assert obj == pytest.approx(-1e6), (
                    f"coilable=2 row expected penalty -1e6, got {obj}"
                )

    def test_idempotency(self) -> None:
        """Running call() twice produces the same objective values."""
        pool = _make_toy_pool()
        block = SupercompressibleObjective()
        block.call(pool)
        first_pass = {
            idx: sample._output_data["objective"]
            for idx, sample in pool.data.items()
        }
        block.call(pool)
        second_pass = {
            idx: sample._output_data["objective"]
            for idx, sample in pool.data.items()
        }
        assert first_pass == second_pass


# =============================================================================
# Tests — agent_main.main() smoke test
# =============================================================================


# Import main() here rather than at module level to avoid running the
# full CLI if importing the module causes side-effects.
from agent_main import main as _agent_main  # noqa: E402


class TestAgentMainSmoke:
    """Smoke tests for ``agent_main.main()`` using a stub agent."""

    def test_returns_expected_keys(self, tmp_path: Path) -> None:
        """main() returns a dict with the four required keys."""
        # Monkey-patch _POOL_DIR inside agent_main to use our toy pool.
        # We do this by subclassing _STUDY_DIR/main to pass project_dir.
        import agent_main as am

        original_pool_dir = am._POOL_DIR

        # Point the module at a tiny temp pool written to disk.
        toy_pool = _make_toy_pool()
        toy_pool_dir = tmp_path / "toy_pool"
        toy_pool_dir.mkdir()
        toy_pool.store(toy_pool_dir)
        am._POOL_DIR = toy_pool_dir

        try:
            result = _agent_main(
                stub=True,
                iterations=2,
                project_dir=tmp_path / "run",
            )
        finally:
            am._POOL_DIR = original_pool_dir

        assert "best_coilable_1" in result
        assert "ceiling" in result
        assert "turn_log_path" in result
        assert "n_evaluated" in result

    def test_turn_log_written(self, tmp_path: Path) -> None:
        """turn_log.jsonl is written after a successful run."""
        import agent_main as am

        original_pool_dir = am._POOL_DIR
        toy_pool = _make_toy_pool()
        toy_pool_dir = tmp_path / "toy_pool"
        toy_pool_dir.mkdir()
        toy_pool.store(toy_pool_dir)
        am._POOL_DIR = toy_pool_dir

        try:
            result = _agent_main(
                stub=True,
                iterations=2,
                project_dir=tmp_path / "run",
            )
        finally:
            am._POOL_DIR = original_pool_dir

        log_path = result["turn_log_path"]
        assert isinstance(log_path, Path)
        assert log_path.exists(), "turn_log.jsonl was not written."
        lines = [ln for ln in log_path.read_text().splitlines() if ln.strip()]
        # 2 iterations × 2 agents = at least 4 turn records.
        assert len(lines) >= 4

    def test_ceiling_is_positive(self, tmp_path: Path) -> None:
        """ceiling should be positive (toy pool has coilable=1 rows)."""
        import agent_main as am

        original_pool_dir = am._POOL_DIR
        toy_pool = _make_toy_pool()
        toy_pool_dir = tmp_path / "toy_pool"
        toy_pool_dir.mkdir()
        toy_pool.store(toy_pool_dir)
        am._POOL_DIR = toy_pool_dir

        try:
            result = _agent_main(
                stub=True,
                iterations=2,
                project_dir=tmp_path / "run",
            )
        finally:
            am._POOL_DIR = original_pool_dir

        assert result["ceiling"] > 0


# =============================================================================
# Tests — deliverable folder
# =============================================================================


class TestDeliverable:
    """Tests for the ``write_deliverable`` run artefact.

    Every test runs ``main(stub=True, iterations=2, ...)`` with the
    20-row toy pool and checks the resulting deliverable folder.
    """

    def _run_with_toy_pool(self, tmp_path: Path) -> dict:
        """Run main() with the toy pool and return the result dict."""
        import agent_main as am

        original_pool_dir = am._POOL_DIR
        toy_pool = _make_toy_pool()
        toy_pool_dir = tmp_path / "toy_pool"
        toy_pool_dir.mkdir()
        toy_pool.store(toy_pool_dir)
        am._POOL_DIR = toy_pool_dir

        try:
            result = _agent_main(
                stub=True,
                iterations=2,
                project_dir=tmp_path / "run",
            )
        finally:
            am._POOL_DIR = original_pool_dir

        return result

    def test_deliverable_folder_exists(self, tmp_path: Path) -> None:
        """Deliverable folder is created after main()."""
        result = self._run_with_toy_pool(tmp_path)
        assert "deliverable_path" in result
        assert result["deliverable_path"].exists()
        assert result["deliverable_path"].is_dir()

    def test_deliverable_contains_required_files(
        self, tmp_path: Path
    ) -> None:
        """solution.md, replicate.py, turn_log.jsonl, experiment_data/
        are all present in the deliverable folder."""
        result = self._run_with_toy_pool(tmp_path)
        deliv = result["deliverable_path"]

        assert (deliv / "solution.md").exists(), "solution.md missing"
        assert (deliv / "replicate.py").exists(), "replicate.py missing"
        assert (
            deliv / "turn_log.jsonl"
        ).exists(), "turn_log.jsonl missing"
        assert (
            deliv / "experiment_data"
        ).is_dir(), "experiment_data/ directory missing"

    def test_solution_md_headings(self, tmp_path: Path) -> None:
        """solution.md contains the required markdown headings."""
        result = self._run_with_toy_pool(tmp_path)
        text = (result["deliverable_path"] / "solution.md").read_text()

        assert "# Solution" in text
        assert "## Run metadata" in text
        # At least one of the optional sections must be present.
        has_comparison = "## Comparison" in text
        has_narration = "## How the solution was reached" in text
        assert has_comparison or has_narration, (
            "solution.md must contain ## Comparison or "
            "## How the solution was reached"
        )

    def test_replicate_py_is_valid_python(self, tmp_path: Path) -> None:
        """replicate.py must be syntactically valid Python."""
        result = self._run_with_toy_pool(tmp_path)
        script_path = result["deliverable_path"] / "replicate.py"
        source = script_path.read_text()
        # compile() raises SyntaxError for invalid Python.
        compile(source, str(script_path), "exec")

    def test_experiment_data_dir_non_empty(
        self, tmp_path: Path
    ) -> None:
        """experiment_data/ inside the deliverable must contain at
        least one file."""
        result = self._run_with_toy_pool(tmp_path)
        exp_data_dir = result["deliverable_path"] / "experiment_data"
        files = list(exp_data_dir.rglob("*"))
        non_dir_files = [f for f in files if f.is_file()]
        assert non_dir_files, (
            "experiment_data/ is empty; expected f3dasm to write "
            "at least one file there."
        )
