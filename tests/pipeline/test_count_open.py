"""Tests for the count_open CLI helper."""

import pytest

from f3dasm import ExperimentData, create_sampler
from f3dasm._src.pipeline.count_open import main
from f3dasm.design import Domain

pytestmark = pytest.mark.smoke


@pytest.fixture
def domain():
    d = Domain()
    d.add_float(name="x0", low=0.0, high=1.0)
    d.add_output(name="y")
    return d


def _store_open_experiments(domain, tmp_path, n: int) -> None:
    """Sample n experiments (all 'open') and persist them to tmp_path."""
    data = ExperimentData(domain=domain).set_project_dir(tmp_path)
    sampler = create_sampler("random", seed=0)
    data = sampler.call(data=data, n_samples=n)
    data.store()


class TestCountOpen:
    def test_prints_open_count(self, domain, tmp_path, capsys):
        _store_open_experiments(domain, tmp_path, 7)

        main([f"--job-dir={tmp_path}"])

        out = capsys.readouterr().out.strip()
        assert out == "7"

    def test_zero_open(self, domain, tmp_path, capsys):
        _store_open_experiments(domain, tmp_path, 0)

        main([f"--job-dir={tmp_path}"])

        assert capsys.readouterr().out.strip() == "0"

    def test_project_subdir(self, domain, tmp_path, capsys):
        sub = tmp_path / "step_a"
        sub.mkdir()
        _store_open_experiments(domain, sub, 3)

        main([f"--job-dir={tmp_path}", "--project-dir=step_a"])

        assert capsys.readouterr().out.strip() == "3"
