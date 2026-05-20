"""Tests for built-in pipeline blocks."""

import json

import pytest

from f3dasm import ExperimentData, create_sampler
from f3dasm._src.pipeline.blocks import CollectArrayResults
from f3dasm.design import Domain

pytestmark = pytest.mark.smoke


@pytest.fixture
def data_with_json(tmp_path):
    """ExperimentData stored to disk with JSON experiment_sample files."""
    domain = Domain()
    domain.add_float("x", 0.0, 1.0)
    domain.add_output("y")
    data = ExperimentData(domain=domain)
    sampler = create_sampler("random", seed=42)
    data = sampler.call(data=data, n_samples=3)
    data.set_project_dir(tmp_path, in_place=True)
    data.store(project_dir=tmp_path)

    # Create experiment_sample JSON files
    sample_dir = tmp_path / "experiment_sample"
    sample_dir.mkdir(parents=True, exist_ok=True)

    for i in range(3):
        sample = data.get_experiment_sample(i)
        sample.store("y", float(i * 10), to_disk=False)
        sample.mark("finished")
        json_data = {
            "input_data": dict(sample._input_data),
            "output_data": dict(sample._output_data),
            "job_status": "FINISHED",
            "project_dir": str(tmp_path),
        }
        with open(sample_dir / f"{i}.json", "w") as f:
            json.dump(json_data, f)

    return data


class TestCollectArrayResults:
    def test_default_cleanup(self):
        block = CollectArrayResults()
        assert block.cleanup is True

    def test_no_cleanup(self):
        block = CollectArrayResults(cleanup=False)
        assert block.cleanup is False

    def test_call_collects_json_files(self, data_with_json, tmp_path):
        block = CollectArrayResults(cleanup=False)
        result = block.call(data=data_with_json)
        assert len(result) == 3
        # experiment_sample dir should still exist
        assert (tmp_path / "experiment_sample").exists()

    def test_call_cleans_up_directory(self, data_with_json, tmp_path):
        block = CollectArrayResults(cleanup=True)
        result = block.call(data=data_with_json)
        assert len(result) == 3
        # experiment_sample dir should be removed
        assert not (tmp_path / "experiment_sample").exists()

    def test_call_no_experiment_sample_dir(self, tmp_path):
        domain = Domain()
        domain.add_float("x", 0.0, 1.0)
        data = ExperimentData(domain=domain)
        data.set_project_dir(tmp_path, in_place=True)

        block = CollectArrayResults(cleanup=True)
        # Should not raise even without experiment_sample dir
        result = block.call(data=data)
        assert result is not None
