"""Tests for AnalysisBase and TaskRegistry (Level 2 typed stores)."""

from __future__ import annotations

import pytest

from f3dasm._src.agentic.stores import (
    AnalysisBase,
    AnalysisNode,
    ContextSlice,
    TaskRegistry,
    TaskStats,
)

__author__ = "Elvis Aguero (elvis_alexander_aguero_vera@brown.edu)"
__credits__ = ["Elvis Aguero"]
__status__ = "Experimental"


# ===========================================================================
# AnalysisBase
# ===========================================================================


def _node(node_id: str, parent_id=None, score=0.5) -> AnalysisNode:
    return AnalysisNode(
        node_id=node_id,
        parent_id=parent_id,
        analysis=f"analysis of {node_id}",
        score=score,
    )


class TestAnalysisBaseAdd:
    def test_add_root(self):
        ab = AnalysisBase()
        ab.add(_node("root"))
        assert len(ab) == 1

    def test_add_child(self):
        ab = AnalysisBase()
        ab.add(_node("root"))
        ab.add(_node("child", parent_id="root"))
        assert len(ab) == 2

    def test_add_duplicate_raises(self):
        ab = AnalysisBase()
        ab.add(_node("root"))
        with pytest.raises(KeyError, match="already in AnalysisBase"):
            ab.add(_node("root"))

    def test_add_missing_parent_raises(self):
        ab = AnalysisBase()
        with pytest.raises(KeyError, match="not in AnalysisBase"):
            ab.add(_node("child", parent_id="nonexistent"))

    def test_contains(self):
        ab = AnalysisBase()
        ab.add(_node("root"))
        assert "root" in ab
        assert "other" not in ab


class TestAnalysisBaseContext:
    def setup_method(self):
        self.ab = AnalysisBase()
        self.ab.add(_node("root"))
        self.ab.add(_node("child_a", parent_id="root"))
        self.ab.add(_node("child_b", parent_id="root"))
        self.ab.add(_node("child_c", parent_id="root"))

    def test_context_missing_raises(self):
        with pytest.raises(KeyError):
            self.ab.context("nonexistent")

    def test_context_returns_context_slice(self):
        sl = self.ab.context("child_a")
        assert isinstance(sl, ContextSlice)

    def test_context_node_itself(self):
        sl = self.ab.context("child_a")
        assert sl.node.node_id == "child_a"

    def test_context_parent(self):
        sl = self.ab.context("child_a")
        assert sl.parent is not None
        assert sl.parent.node_id == "root"

    def test_context_root_has_no_parent(self):
        sl = self.ab.context("root")
        assert sl.parent is None

    def test_context_siblings(self):
        sl = self.ab.context("child_a")
        sibling_ids = {n.node_id for n in sl.siblings}
        assert sibling_ids == {"child_b", "child_c"}
        assert "child_a" not in sibling_ids

    def test_context_root_has_no_siblings(self):
        sl = self.ab.context("root")
        assert sl.siblings == []

    def test_context_uncles(self):
        ab = AnalysisBase()
        ab.add(_node("gp"))
        ab.add(_node("parent_a", parent_id="gp"))
        ab.add(_node("parent_b", parent_id="gp"))
        ab.add(_node("child", parent_id="parent_a"))
        ab.add(_node("uncle_child", parent_id="parent_b"))

        sl = ab.context("child")
        uncle_ids = {n.node_id for n in sl.uncles}
        assert "uncle_child" in uncle_ids

    def test_context_no_uncles_if_parent_is_root(self):
        sl = self.ab.context("child_a")
        assert sl.uncles == []

    def test_context_include_parent_only(self):
        sl = self.ab.context("child_a", include=("parent",))
        assert sl.parent is not None
        assert sl.siblings == []
        assert sl.uncles == []

    def test_context_include_siblings_only(self):
        sl = self.ab.context("child_a", include=("siblings",))
        assert sl.parent is None
        assert len(sl.siblings) == 2
        assert sl.uncles == []

    def test_context_unknown_include_raises(self):
        with pytest.raises(ValueError, match="unknown include values"):
            self.ab.context("child_a", include=("parents",))


# ===========================================================================
# TaskRegistry
# ===========================================================================


class TestTaskStats:
    def test_success_rate_zero_attempts(self):
        s = TaskStats(task_name="t", task_text="txt")
        assert s.success_rate == 0.0

    def test_success_rate_computation(self):
        s = TaskStats(task_name="t", task_text="txt", attempts=4, successes=3)
        assert s.success_rate == pytest.approx(0.75)

    def test_task_stats_is_frozen(self):
        s = TaskStats(task_name="t", task_text="txt")
        with pytest.raises((AttributeError, TypeError)):
            s.attempts = 5  # type: ignore[misc]

    def test_task_stats_has_no_is_default(self):
        s = TaskStats(task_name="t", task_text="txt")
        assert not hasattr(s, "is_default")


class TestTaskRegistryRegister:
    def test_register_and_contains(self):
        reg = TaskRegistry()
        reg.register("lhs", "Latin hypercube sample")
        assert "lhs" in reg

    def test_register_updates_existing(self):
        reg = TaskRegistry()
        reg.register("lhs", "old text")
        reg.register("lhs", "new text")
        assert reg["lhs"].task_text == "new text"
        assert len(reg) == 1

    def test_register_default_never_evicted(self):
        reg = TaskRegistry(max_tasks=2)
        reg.register("default_op", "default", is_default=True)
        reg.register("non_default", "non default")
        reg.register("another", "another non default")
        assert "default_op" in reg
        assert len(reg) == 2

    def test_register_evicts_lowest_success_rate(self):
        reg = TaskRegistry(max_tasks=2)
        reg.register("op_a", "text a")
        reg.register("op_b", "text b")
        reg.update("op_a", improved=True)
        reg.register("op_c", "text c")
        assert "op_a" in reg
        assert "op_c" in reg
        assert "op_b" not in reg

    def test_register_full_all_defaults_raises(self):
        reg = TaskRegistry(max_tasks=1)
        reg.register("d", "default", is_default=True)
        with pytest.raises(RuntimeError, match="all entries are defaults"):
            reg.register("new", "new text")


class TestTaskRegistryUpdate:
    def test_update_increments_attempts(self):
        reg = TaskRegistry()
        reg.register("op", "text")
        reg.update("op", improved=True)
        assert reg["op"].attempts == 1
        assert reg["op"].successes == 1

    def test_update_increments_only_attempts_on_failure(self):
        reg = TaskRegistry()
        reg.register("op", "text")
        reg.update("op", improved=False)
        assert reg["op"].attempts == 1
        assert reg["op"].successes == 0

    def test_update_missing_raises(self):
        reg = TaskRegistry()
        with pytest.raises(KeyError):
            reg.update("nonexistent", improved=True)


class TestTaskRegistryTopK:
    def test_top_k_empty(self):
        reg = TaskRegistry()
        assert reg.top_k(5) == []

    def test_top_k_sorted_by_success_rate(self):
        reg = TaskRegistry()
        reg.register("a", "a")
        reg.register("b", "b")
        reg.register("c", "c")
        # a: 1 attempt, 1 success → rate 1.0
        reg.update("a", improved=True)
        # b: 3 attempts, 2 successes → rate 0.667
        reg.update("b", improved=True)
        reg.update("b", improved=True)
        reg.update("b", improved=False)
        top = reg.top_k(2)
        assert top[0].task_name == "a"
        assert top[1].task_name == "b"

    def test_top_k_returns_task_stats(self):
        reg = TaskRegistry()
        reg.register("op", "text")
        top = reg.top_k(1)
        assert isinstance(top[0], TaskStats)

    def test_top_k_capped(self):
        reg = TaskRegistry()
        for i in range(5):
            reg.register(f"op{i}", f"text {i}")
        assert len(reg.top_k(3)) == 3

    def test_top_k_deterministic_tie(self):
        reg = TaskRegistry()
        reg.register("b_op", "b")
        reg.register("a_op", "a")
        top = reg.top_k(10)
        names = [e.task_name for e in top]
        assert names == sorted(names)


class TestTaskRegistryIteration:
    def test_len(self):
        reg = TaskRegistry()
        reg.register("a", "a")
        reg.register("b", "b")
        assert len(reg) == 2

    def test_iter_yields_task_stats(self):
        reg = TaskRegistry()
        reg.register("a", "a")
        entries = list(reg)
        assert len(entries) == 1
        assert isinstance(entries[0], TaskStats)
        assert entries[0].task_name == "a"

    def test_default_max_tasks_is_20(self):
        reg = TaskRegistry()
        for i in range(20):
            reg.register(f"op{i}", f"text {i}")
        assert len(reg) == 20
        reg.register("op_extra", "extra")
        assert len(reg) == 20
