"""Tests for core Block, DataGenerator, Optimizer ABCs."""

from unittest.mock import patch

import pytest

from f3dasm import ExperimentData
from f3dasm._src.core import (
    Block,
    ChainedBlock,
    DataGenerator,
    Optimizer,
    datagenerator,
)
from f3dasm._src.experimentsample import ExperimentSample
from f3dasm.design import Domain

pytestmark = pytest.mark.smoke


# ======================= Block ABC =======================


def test_block_is_abstract():
    """Block cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Block()


def test_block_subclass_must_implement_call():
    """A subclass that doesn't implement call() can't be instantiated."""

    class IncompleteBlock(Block):
        pass

    with pytest.raises(TypeError):
        IncompleteBlock()


def test_block_subclass_with_call():
    """A proper subclass can be instantiated and called."""

    class MyBlock(Block):
        def call(self, data, **kwargs):
            return data

    block = MyBlock()
    data = ExperimentData()
    result = block.call(data)
    assert result is data


def test_block_arm_default_noop():
    """arm() is a no-op by default."""

    class MyBlock(Block):
        def call(self, data, **kwargs):
            return data

    block = MyBlock()
    data = ExperimentData()
    # Should not raise
    block.arm(data)


# ======================= DataGenerator ABC =======================


def test_datagenerator_execute_raises_not_implemented():
    """DataGenerator.execute raises NotImplementedError by default."""
    gen = DataGenerator()
    sample = ExperimentSample(_input_data={"x0": 1.0})
    with pytest.raises(NotImplementedError):
        gen.execute(sample)


def test_datagenerator_subclass():
    """A proper DataGenerator subclass works."""

    class MyGen(DataGenerator):
        def execute(self, experiment_sample, **kwargs):
            experiment_sample.store("y", 0.0)
            return experiment_sample

    gen = MyGen()
    sample = ExperimentSample(_input_data={"x0": 1.0})
    result = gen.execute(sample)
    assert result.output_data["y"] == 0.0


def test_datagenerator_call_invalid_mode():
    """Invalid parallelization mode raises ValueError."""

    class MyGen(DataGenerator):
        def execute(self, experiment_sample, **kwargs):
            return experiment_sample

    gen = MyGen()
    data = ExperimentData(input_data=[{"x0": 1.0}])
    with pytest.raises(ValueError, match="Invalid parallelization mode"):
        gen.call(data, mode="invalid_mode")


# ======================= Optimizer ABC =======================


def test_optimizer_is_abstract():
    """Optimizer cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Optimizer()


def test_optimizer_subclass_must_implement_arm_and_call():
    """Optimizer subclass must implement both arm and call."""

    class IncompleteOptimizer(Optimizer):
        def arm(self, data, data_generator, input_name, output_name):
            pass

    with pytest.raises(TypeError):
        IncompleteOptimizer()


# ======================= datagenerator decorator =======================


def test_datagenerator_decorator():
    """The datagenerator decorator creates a DataGenerator."""

    @datagenerator(output_names="y")
    def f(x0):
        return x0**2

    assert isinstance(f, DataGenerator)


def test_datagenerator_decorator_multiple_outputs():
    """Multiple output names are supported."""

    @datagenerator(output_names=["y0", "y1"])
    def f(x0):
        return x0, x0**2

    sample = ExperimentSample(_input_data={"x0": 3.0})
    result = f.execute(sample)
    assert result.output_data["y0"] == 3.0
    assert result.output_data["y1"] == 9.0


def test_datagenerator_decorator_no_output_names_raises():
    """Empty output_names should raise ValueError."""
    with pytest.raises(ValueError):

        @datagenerator(output_names=[])
        def f(x0):
            return x0


def test_datagenerator_decorator_sequential_execution():
    """datagenerator can execute sequentially on ExperimentData."""

    @datagenerator(output_names="y")
    def f(x0):
        return x0**2

    data = ExperimentData(input_data=[{"x0": 2.0}, {"x0": 3.0}])
    result = f.call(data)
    _, df_out = result.to_pandas()
    assert df_out["y"].iloc[0] == 4.0
    assert df_out["y"].iloc[1] == 9.0


# ======================= Block >> chaining =======================


class TestBlockChaining:
    def test_rshift_two_blocks(self):
        class AddOne(Block):
            def call(self, data, **kwargs):
                return data

        result = AddOne() >> AddOne()
        assert isinstance(result, ChainedBlock)
        assert len(result.blocks) == 2

    def test_rshift_chained_then_block(self):
        class Noop(Block):
            def call(self, data, **kwargs):
                return data

        chained = ChainedBlock([Noop(), Noop()])
        result = chained >> Noop()
        assert isinstance(result, ChainedBlock)
        assert len(result.blocks) == 3

    def test_rshift_chained_then_chained(self):
        class Noop(Block):
            def call(self, data, **kwargs):
                return data

        c1 = ChainedBlock([Noop()])
        c2 = ChainedBlock([Noop(), Noop()])
        result = c1 >> c2
        assert isinstance(result, ChainedBlock)
        assert len(result.blocks) == 3

    def test_chained_call_runs_all(self):
        class Counter(Block):
            count = 0

            def call(self, data, **kwargs):
                Counter.count += 1
                return data

        Counter.count = 0
        chained = ChainedBlock([Counter(), Counter(), Counter()])
        chained.call(data=ExperimentData())
        assert Counter.count == 3

    def test_chained_arm_arms_all(self):
        armed = []

        class Armable(Block):
            def __init__(self, name):
                self.name = name

            def arm(self, data):
                armed.append(self.name)

            def call(self, data, **kwargs):
                return data

        chained = ChainedBlock([Armable("a"), Armable("b")])
        chained.arm(data=ExperimentData())
        assert armed == ["a", "b"]


# ======================= DataGenerator mode dispatch =======================


class TestDataGeneratorModes:
    def _make_gen(self):
        class SimpleGen(DataGenerator):
            def execute(self, experiment_sample, **kwargs):
                experiment_sample.store("y", 1.0, to_disk=False)
                return experiment_sample

        return SimpleGen()

    def test_call_parallel_mode(self):
        gen = self._make_gen()
        data = ExperimentData(input_data=[{"x0": 1.0}])
        result = gen.call(data, mode="parallel", nodes=1)
        assert len(result) == 1

    def test_call_cluster_mode(self, tmp_path):
        gen = self._make_gen()
        data = ExperimentData(input_data=[{"x0": 1.0}])
        data.set_project_dir(tmp_path, in_place=True)
        data.store(project_dir=tmp_path)
        result = gen.call(data, mode="cluster")
        # evaluate_cluster returns None but data is modified on disk
        assert result is None

    def test_call_cluster_array_mode(self, tmp_path):
        gen = self._make_gen()
        data = ExperimentData(input_data=[{"x0": 1.0}])
        data.set_project_dir(tmp_path, in_place=True)
        for es in data.data.values():
            es.project_dir = tmp_path
        result = gen.call(data, mode="cluster_array", job_number=0)
        assert result is None


# ======================= datagenerator decorator edge cases =======================


def test_datagenerator_decorator_with_defaults():
    """Decorator should handle functions with default parameters."""

    @datagenerator(output_names="y")
    def f(x0, scale=2.0):
        return x0 * scale

    sample = ExperimentSample(_input_data={"x0": 3.0})
    result = f.execute(sample)
    assert result.output_data["y"] == 6.0


def test_datagenerator_decorator_with_domain():
    """Decorator with domain should propagate parameter metadata."""
    domain = Domain()
    domain.add_float("x0", 0.0, 1.0)
    domain.add_output("y")

    @datagenerator(output_names="y", domain=domain)
    def f(x0):
        return x0**2

    assert isinstance(f, DataGenerator)
