"""Lookup-based data generator.

The ``LookupDataGenerator`` evaluates a proposed candidate by projecting it
onto the nearest entry of a pre-computed pool and returning that entry's
output columns. It is the Level-1 evaluator used by the agentic-f3dasm MVP
when no live simulator (e.g. Abaqus) is available.

The generator is problem-agnostic: it makes no assumptions about which
columns are inputs or outputs beyond what the caller specifies at
construction time.
"""
#                                                                       Modules
# =============================================================================
from __future__ import annotations

# Standard
from copy import deepcopy
from typing import Optional

# Third-party
import numpy as np

# Local
from ..core import DataGenerator
from ..experimentdata import ExperimentData
from ..experimentsample import ExperimentSample, JobStatus

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Elvis Aguero (elvis_alexander_aguero_vera@brown.edu)"
__credits__ = ["Elvis Aguero"]
__status__ = "Experimental"
# =============================================================================


class LookupDataGenerator(DataGenerator):
    """Evaluate candidates by nearest-neighbour lookup against a fixed pool.

    For every ``execute`` call the generator finds the pool row whose
    normalised input vector is closest (L2 distance over min-max-scaled
    coordinates) to the candidate and copies that row's output columns
    onto the sample. The mechanism is the canonical Level-1 evaluator for
    agentic-f3dasm — the agent never proposes coordinates that the pool
    knows about ahead of time, but every proposal can be mapped to a pool
    row deterministically.

    Parameters
    ----------
    pool : ExperimentData
        The pre-computed dataset to look up against. Must already carry
        finished output columns for every row.
    input_columns : list of str
        Names of the input columns (already present on every pool row)
        that contribute to the L2 distance.
    output_columns : list of str, optional
        Names of the output columns to copy from the matched pool row.
        If ``None``, every key in the matched row's ``_output_data`` dict
        is copied.

    Attributes
    ----------
    pool : ExperimentData
        The pool the generator was constructed with (kept by reference for
        downstream introspection; never mutated).
    seen_indices : set of int
        Pool indices that have already been returned during this run; the
        set is reset by :meth:`reset_seen`.

    Notes
    -----
    The deep-copy of pool output data on every call guarantees that the
    pool is never mutated by downstream code that edits the returned
    sample. The generator deliberately does NOT raise on a repeated pool
    hit — exhaustion of one region is not exhaustion of the search.
    Strategies can poll :meth:`consume_repeats` after a batch to surface a
    warning to the agent.
    """

    def __init__(
        self,
        pool: ExperimentData,
        input_columns: list[str],
        output_columns: Optional[list[str]] = None,
    ) -> None:
        self.pool = pool
        self.input_columns = list(input_columns)
        self.output_columns = (
            None if output_columns is None else list(output_columns)
        )

        # Materialise the pool's input vectors and bounds once so every
        # execute() is a constant-time normalisation + linear scan.
        pool_inputs = np.asarray(
            [
                [row._input_data[col] for col in self.input_columns]
                for row in pool.data.values()
            ],
            dtype=float,
        )
        self._pool_indices: list[int] = list(pool.data.keys())
        self._pool_inputs = pool_inputs

        # Per-column min/max for L2 normalisation. A column with zero
        # spread (all values equal) gets a unit denominator so the
        # contribution to L2 collapses to 0 rather than NaN.
        col_min = pool_inputs.min(axis=0)
        col_max = pool_inputs.max(axis=0)
        spread = col_max - col_min
        spread[spread == 0.0] = 1.0
        self._col_min = col_min
        self._col_spread = spread
        self._pool_normalised = (pool_inputs - col_min) / spread

        # Provenance for warning surfaces consumed by strategy adapters.
        self.seen_indices: set[int] = set()
        self._repeats: int = 0

    # --------------------------------------------------------- public API

    def execute(
        self, experiment_sample: ExperimentSample, **kwargs
    ) -> ExperimentSample:
        """Look up the nearest pool row and copy its outputs onto the sample.

        Parameters
        ----------
        experiment_sample : ExperimentSample
            Sample whose input data carries values for every column in
            :attr:`input_columns`.
        **kwargs : dict
            Unused; present to match the ABC signature.

        Returns
        -------
        ExperimentSample
            The same sample object, with output data filled and the job
            status marked as finished.

        Raises
        ------
        KeyError
            If the sample is missing one of the configured input columns.
        """

        # Vectorised L2 over the normalised pool.
        candidate = np.asarray(
            [experiment_sample._input_data[col] for col in self.input_columns],
            dtype=float,
        )
        normalised_candidate = (candidate - self._col_min) / self._col_spread
        distances = np.linalg.norm(
            self._pool_normalised - normalised_candidate, axis=1
        )
        nearest_local_idx = int(np.argmin(distances))
        nearest_pool_idx = self._pool_indices[nearest_local_idx]

        matched_row = self.pool.data[nearest_pool_idx]
        # Deep-copy guards against the agent mutating the pool via the
        # returned sample.
        source_outputs = deepcopy(matched_row._output_data)
        if self.output_columns is None:
            outputs_to_copy = source_outputs
        else:
            outputs_to_copy = {
                key: source_outputs[key]
                for key in self.output_columns
                if key in source_outputs
            }

        experiment_sample._output_data.update(outputs_to_copy)
        experiment_sample.job_status = JobStatus.FINISHED

        if nearest_pool_idx in self.seen_indices:
            self._repeats += 1
        else:
            self.seen_indices.add(nearest_pool_idx)

        return experiment_sample

    def consume_repeats(self) -> int:
        """Return and reset the repeat counter.

        Strategy adapters call this after a batch to surface a pool
        exhaustion warning to the agent.

        Returns
        -------
        int
            Number of times ``execute`` returned a previously-seen pool
            row since the last call to this method.
        """

        repeats = self._repeats
        self._repeats = 0
        return repeats

    def reset_seen(self) -> None:
        """Forget which pool indices have already been returned.

        Useful for fresh runs that reuse the same pool instance.
        """

        self.seen_indices.clear()
        self._repeats = 0


__all__ = ["LookupDataGenerator"]
