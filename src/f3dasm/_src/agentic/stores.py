"""Typed shared-state stores for Level 2 agentic-f3dasm topologies.

Two concrete stores support the reference topologies described in
``docs/specs/architecture.md``:

- :class:`AnalysisBase` — hierarchical per-solution analysis store.
  ``get(node_id)`` returns the parent, sibling, and uncle reports for a
  given solution node, bounding context while preserving lineage.

- :class:`TaskRegistry` — PABLO-inspired capped operator registry.
  Each entry tracks ``attempts``, ``successes``, and ``success_rate``
  so a Planner can weight operator selection toward historically
  effective tasks.
"""

#                                                                       Modules
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Elvis Aguero (elvis_alexander_aguero_vera@brown.edu)"
__credits__ = ["Elvis Aguero"]
__status__ = "Experimental"
# =============================================================================

__all__ = [
    "AnalysisBase",
    "AnalysisNode",
    "AnalysisSlice",
    "MAX_TASKS",
    "RegistryEntry",
    "TaskRegistry",
]

# ---------------------------------------------------------------------------
# AnalysisBase
# ---------------------------------------------------------------------------

MAX_TASKS: int = 20
"""Default cap on the number of entries in a :class:`TaskRegistry`."""


@dataclass
class AnalysisNode:
    """One node in the solution tree held by :class:`AnalysisBase`.

    Parameters
    ----------
    node_id : str
        Unique identifier for this solution.
    parent_id : str or None
        Identifier of the parent solution, or ``None`` for root nodes.
    analysis : str
        Text summary produced by the ResultAnalyst (or equivalent role).
    score : float
        Numeric score for this solution.
    metadata : dict
        Optional channel-specific additions (plot paths, extra metrics, …).
    """

    node_id: str
    parent_id: str | None
    analysis: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisSlice:
    """Bounded comparative context returned by :meth:`AnalysisBase.get`.

    Parameters
    ----------
    node : AnalysisNode
        The queried node itself.
    parent : AnalysisNode or None
        The parent node (lineage), or ``None`` if the node is a root.
    siblings : list[AnalysisNode]
        Other children of the same parent (what else was tried at the
        same generation), excluding the queried node itself.
    uncles : list[AnalysisNode]
        Children of the parent's siblings (what the parent's competitors
        produced).
    """

    node: AnalysisNode
    parent: AnalysisNode | None
    siblings: list[AnalysisNode]
    uncles: list[AnalysisNode]


class AnalysisBase:
    """Hierarchical store of per-solution analysis reports.

    Nodes form a tree: each node has at most one parent and zero or more
    children.  :meth:`get` returns the *parent + siblings + uncles* slice
    that gives a Critic or SelectorEnsemble the minimum comparative
    context for a well-grounded judgment without reading the full tree.

    Examples
    --------
    >>> ab = AnalysisBase()
    >>> ab.add(AnalysisNode("root", None, "baseline design", 0.5))
    >>> ab.add(AnalysisNode("child_a", "root", "improved v1", 0.7))
    >>> ab.add(AnalysisNode("child_b", "root", "improved v2", 0.6))
    >>> sl = ab.get("child_a")
    >>> sl.parent.node_id
    'root'
    >>> [n.node_id for n in sl.siblings]
    ['child_b']
    """

    def __init__(self) -> None:
        self._nodes: dict[str, AnalysisNode] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, node: AnalysisNode) -> None:
        """Add a node to the store.

        Parameters
        ----------
        node : AnalysisNode
            The node to add.  Its ``node_id`` must be unique; its
            ``parent_id`` (if not ``None``) must already be present.

        Raises
        ------
        KeyError
            If ``node.node_id`` is already registered, or if
            ``node.parent_id`` is not ``None`` and is not registered.
        """
        if node.node_id in self._nodes:
            raise KeyError(
                f"node_id {node.node_id!r} already in AnalysisBase."
            )
        if node.parent_id is not None and node.parent_id not in self._nodes:
            raise KeyError(
                f"parent_id {node.parent_id!r} not in AnalysisBase."
            )
        self._nodes[node.node_id] = node

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get(self, node_id: str) -> AnalysisSlice:
        """Return the bounded comparative context for ``node_id``.

        Parameters
        ----------
        node_id : str
            The node to query.

        Returns
        -------
        AnalysisSlice
            Contains the node itself, its parent (or ``None``), its
            siblings (other children of the same parent), and its uncles
            (children of the parent's siblings).

        Raises
        ------
        KeyError
            If ``node_id`` is not in the store.
        """
        node = self._nodes[node_id]
        parent: AnalysisNode | None = None
        siblings: list[AnalysisNode] = []
        uncles: list[AnalysisNode] = []

        if node.parent_id is not None:
            parent = self._nodes[node.parent_id]
            siblings = [
                n
                for n in self._nodes.values()
                if n.parent_id == node.parent_id and n.node_id != node_id
            ]
            parent_siblings = [
                n
                for n in self._nodes.values()
                if parent.parent_id is not None
                and n.parent_id == parent.parent_id
                and n.node_id != node.parent_id
            ]
            uncles = [
                n
                for n in self._nodes.values()
                if n.parent_id in {ps.node_id for ps in parent_siblings}
            ]

        return AnalysisSlice(
            node=node,
            parent=parent,
            siblings=siblings,
            uncles=uncles,
        )

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node_id: str) -> bool:
        return node_id in self._nodes


# ---------------------------------------------------------------------------
# TaskRegistry
# ---------------------------------------------------------------------------


@dataclass
class RegistryEntry:
    """One entry in a :class:`TaskRegistry`.

    Parameters
    ----------
    task_name : str
        Short name identifying the operator.
    task_text : str
        Full natural-language description of the task.
    attempts : int
        Total number of times this task has been dispatched.
    successes : int
        Number of dispatches judged successful.
    is_default : bool
        Default operators are never pruned when the registry is at
        capacity.
    """

    task_name: str
    task_text: str
    attempts: int = 0
    successes: int = 0
    is_default: bool = False

    @property
    def success_rate(self) -> float:
        """Fraction of attempts that succeeded (0.0 if no attempts)."""
        return self.successes / self.attempts if self.attempts > 0 else 0.0


class TaskRegistry:
    """Capped registry of reusable operator descriptions.

    The Planner queries the registry to select operators for Worker
    delegations; Workers update ``attempts`` and ``successes`` after each
    task.  The registry is capped at ``max_tasks`` entries; default
    entries (``is_default=True``) are never pruned.

    Parameters
    ----------
    max_tasks : int
        Maximum number of entries.  Default is :data:`MAX_TASKS` (20).

    Examples
    --------
    >>> reg = TaskRegistry()
    >>> reg.register("lhs_sample", "Run a Latin hypercube sample of N points")
    >>> reg.update("lhs_sample", improved=True)
    >>> reg.update("lhs_sample", improved=False)
    >>> reg["lhs_sample"].success_rate
    0.5
    """

    def __init__(self, max_tasks: int = MAX_TASKS) -> None:
        self._entries: dict[str, RegistryEntry] = {}
        self._max_tasks = max_tasks

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def register(
        self,
        task_name: str,
        task_text: str,
        *,
        is_default: bool = False,
    ) -> None:
        """Add or replace a task entry.

        When the registry is at capacity, the non-default entry with the
        lowest ``success_rate`` is evicted to make room.  If all
        non-default entries have the same success rate, the oldest (by
        insertion order) is evicted.  If the registry is full and all
        entries are defaults, raises :class:`RuntimeError`.

        Parameters
        ----------
        task_name : str
            Short identifier for the operator.
        task_text : str
            Full natural-language description.
        is_default : bool
            Whether this entry is protected from eviction.

        Raises
        ------
        RuntimeError
            If the registry is at capacity and all existing entries are
            defaults.
        """
        if task_name in self._entries:
            self._entries[task_name].task_text = task_text
            self._entries[task_name].is_default = (
                self._entries[task_name].is_default or is_default
            )
            return

        if len(self._entries) >= self._max_tasks:
            evictable = [
                e for e in self._entries.values() if not e.is_default
            ]
            if not evictable:
                raise RuntimeError(
                    f"TaskRegistry is full ({self._max_tasks} entries) "
                    "and all entries are defaults."
                )
            worst = min(evictable, key=lambda e: e.success_rate)
            del self._entries[worst.task_name]

        self._entries[task_name] = RegistryEntry(
            task_name=task_name,
            task_text=task_text,
            is_default=is_default,
        )

    def update(self, task_name: str, *, improved: bool) -> None:
        """Record the outcome of one task dispatch.

        Parameters
        ----------
        task_name : str
            The task that was dispatched.
        improved : bool
            Whether the dispatch was judged successful.

        Raises
        ------
        KeyError
            If ``task_name`` is not in the registry.
        """
        entry = self._entries[task_name]
        entry.attempts += 1
        if improved:
            entry.successes += 1

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def top_k(self, k: int) -> list[RegistryEntry]:
        """Return up to ``k`` entries sorted by descending success rate.

        Parameters
        ----------
        k : int
            Maximum number of entries to return.

        Returns
        -------
        list[RegistryEntry]
            Entries sorted by ``success_rate`` descending, then by
            ``task_name`` ascending for deterministic ties.
        """
        sorted_entries = sorted(
            self._entries.values(),
            key=lambda e: (-e.success_rate, e.task_name),
        )
        return sorted_entries[:k]

    def __getitem__(self, task_name: str) -> RegistryEntry:
        return self._entries[task_name]

    def __contains__(self, task_name: str) -> bool:
        return task_name in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries.values())
