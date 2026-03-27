"""Extended tests for ExperimentSample: repr, from_json, copy, status."""

import json
from pathlib import Path

import numpy as np
import pytest

from f3dasm import ExperimentSample
from f3dasm._src.errors import DecodeError
from f3dasm._src.experimentsample import JobStatus

pytestmark = pytest.mark.smoke


class TestRepr:
    def test_repr_contains_data(self):
        sample = ExperimentSample(
            _input_data={"x": 1}, _output_data={"y": 2}
        )
        r = repr(sample)
        assert "ExperimentSample" in r
        assert "x" in r
        assert "y" in r
        assert "FINISHED" in r

    def test_repr_empty(self):
        sample = ExperimentSample()
        r = repr(sample)
        assert "OPEN" in r


class TestJobStatusStr:
    def test_str_conversion(self):
        assert str(JobStatus.OPEN) == "OPEN"
        assert str(JobStatus.IN_PROGRESS) == "IN_PROGRESS"
        assert str(JobStatus.FINISHED) == "FINISHED"
        assert str(JobStatus.ERROR) == "ERROR"


class TestJobStatusFromString:
    def test_valid_status_strings(self):
        sample = ExperimentSample(job_status="OPEN")
        assert sample.job_status == JobStatus.OPEN

        sample = ExperimentSample(job_status="FINISHED")
        assert sample.job_status == JobStatus.FINISHED

    def test_invalid_status_string(self):
        with pytest.raises(DecodeError):
            ExperimentSample(job_status="INVALID_STATUS")


class TestCopy:
    def test_copy_creates_independent_instance(self):
        original = ExperimentSample(
            _input_data={"x": 1}, _output_data={"y": 2}
        )
        copy = original._copy()
        assert copy == original
        copy._input_data["x"] = 999
        assert original._input_data["x"] == 1

    def test_copy_preserves_status(self):
        original = ExperimentSample(job_status="IN_PROGRESS")
        copy = original._copy()
        assert copy.job_status == JobStatus.IN_PROGRESS


class TestFromJson:
    def test_from_json(self, tmp_path):
        json_data = {
            "input_data": {"x": 1.0},
            "output_data": {"y": 2.0},
            "job_status": "FINISHED",
            "project_dir": str(tmp_path),
        }
        json_path = tmp_path / "sample.json"
        json_path.write_text(json.dumps(json_data))
        sample = ExperimentSample.from_json(json_path)
        assert sample.input_data == {"x": 1.0}
        assert sample.output_data == {"y": 2.0}
        assert sample.job_status == JobStatus.FINISHED


class TestStoreAsJson:
    def test_store_as_json(self, tmp_path):
        sample = ExperimentSample(
            _input_data={"x": 1.0},
            _output_data={"y": 2.0},
            job_status="FINISHED",
            project_dir=tmp_path,
        )
        sample.store_as_json(idx=0)
        json_path = tmp_path / "experiment_sample" / "0.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["input_data"]["x"] == 1.0


class TestNoneData:
    def test_none_input_defaults_to_empty(self):
        sample = ExperimentSample(_input_data=None, _output_data=None)
        assert sample._input_data == {}
        assert sample._output_data == {}

    def test_output_data_infers_finished(self):
        sample = ExperimentSample(
            _input_data={"x": 1}, _output_data={"y": 2}
        )
        assert sample.job_status == JobStatus.FINISHED

    def test_no_output_data_infers_open(self):
        sample = ExperimentSample(_input_data={"x": 1})
        assert sample.job_status == JobStatus.OPEN
