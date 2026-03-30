"""Extended tests for ExperimentData: iteration, selection, move operations."""

import numpy as np
import pytest

from f3dasm import ExperimentData, ExperimentSample, create_sampler
from f3dasm.design import Domain

pytestmark = pytest.mark.smoke

SEED = 42


@pytest.fixture
def domain():
    d = Domain()
    d.add_float(name="x0", low=0.0, high=1.0)
    d.add_float(name="x1", low=0.0, high=1.0)
    d.add_output(name="y")
    return d


@pytest.fixture
def data(domain):
    d = ExperimentData(domain=domain)
    sampler = create_sampler("random", seed=SEED)
    d = sampler.call(data=d, n_samples=5)
    return d


class TestIteration:
    def test_iter(self, data):
        items = list(data)
        assert len(items) == 5
        for idx, sample in items:
            assert isinstance(idx, int)
            assert isinstance(sample, ExperimentSample)

    def test_len(self, data):
        assert len(data) == 5

    def test_getitem(self, data):
        result = data[0]
        # __getitem__ returns an ExperimentData slice, not ExperimentSample
        assert isinstance(result, ExperimentData)
        assert len(result) == 1


class TestSelectParameter:
    def test_select_input_parameter(self, data):
        selected = data.select_parameter("x0")
        assert len(selected) == 5
        assert "x0" in selected._domain.input_space
        assert "x1" not in selected._domain.input_space

    def test_select_output_parameter(self, data):
        # First add output data
        from f3dasm import datagenerator

        @datagenerator(output_names="y")
        def f(*args, **kwargs):
            return 0.0

        data_with_output = f.call(data=data)
        selected = data_with_output.select_parameter("y")
        assert "y" in selected._domain.output_space

    def test_select_nonexistent_parameter(self, data):
        with pytest.raises(KeyError, match="not found"):
            data.select_parameter("nonexistent")


class TestMoveToOutput:
    def test_move_to_output(self, data):
        result = data.move_to_output("x0")
        assert "x0" in result._domain.output_space
        assert "x0" not in result._domain.input_space
        # Original should be unchanged
        assert "x0" in data._domain.input_space

    def test_move_to_output_in_place(self, data):
        result = data.move_to_output("x0", in_place=True)
        assert result is None
        assert "x0" in data._domain.output_space

    def test_move_to_output_nonexistent(self, data):
        with pytest.raises(KeyError, match="not found in input"):
            data.move_to_output("nonexistent")


class TestMoveToInput:
    def test_move_to_input(self, data):
        from f3dasm import datagenerator

        @datagenerator(output_names="y")
        def f(*args, **kwargs):
            return 0.0

        data_with_output = f.call(data=data)
        result = data_with_output.move_to_input("y")
        assert "y" in result._domain.input_space
        assert "y" not in result._domain.output_space

    def test_move_to_input_nonexistent(self, data):
        with pytest.raises(KeyError, match="not found in output"):
            data.move_to_input("nonexistent")


class TestSelectWithStatus:
    def test_select_open(self, data):
        open_data = data.select_with_status("open")
        assert len(open_data) == 5

    def test_select_finished_empty(self, data):
        finished_data = data.select_with_status("finished")
        assert len(finished_data) == 0


class TestToNumpy:
    def test_to_numpy(self, data):
        from f3dasm import datagenerator

        @datagenerator(output_names="y")
        def f(*args, **kwargs):
            return 0.0

        data_with_output = f.call(data=data)
        input_arr, output_arr = data_with_output.to_numpy()
        assert isinstance(input_arr, np.ndarray)
        assert isinstance(output_arr, np.ndarray)
        assert input_arr.shape == (5, 2)
        assert output_arr.shape == (5, 1)


class TestAddition:
    def test_add_experiment_data(self, domain):
        d1 = ExperimentData(domain=domain)
        s1 = create_sampler("random", seed=1)
        d1 = s1.call(data=d1, n_samples=3)
        d2 = ExperimentData(domain=domain)
        s2 = create_sampler("random", seed=2)
        d2 = s2.call(data=d2, n_samples=2)
        combined = d1 + d2
        assert len(combined) == 5


class TestStoreToDisk:
    def test_store_and_load(self, data, tmp_path):
        data = data.set_project_dir(tmp_path)
        data.store(tmp_path)
        loaded = ExperimentData.from_file(project_dir=tmp_path)
        assert len(loaded) == len(data)


class TestFromFile:
    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ExperimentData.from_file(project_dir=tmp_path / "nonexistent")
