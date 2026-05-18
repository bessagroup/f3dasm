"""Tests for the ``LookupDataGenerator``."""
#                                                                       Modules
# =============================================================================
from __future__ import annotations

# Local
from f3dasm import ExperimentData
from f3dasm._src.agentic.lookup import LookupDataGenerator
from f3dasm._src.experimentsample import ExperimentSample
from f3dasm.design import Domain

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Elvis Aguero (elvis_alexander_aguero_vera@brown.edu)"
__credits__ = ["Elvis Aguero"]
__status__ = "Experimental"
# =============================================================================


def _make_pool() -> ExperimentData:
    """Build a tiny 3-row, 2-input, 1-output pool for the unit tests."""

    domain = Domain()
    domain.add_float("x0", low=0.0, high=10.0)
    domain.add_float("x1", low=0.0, high=10.0)
    domain.add_output("y")
    return ExperimentData(
        domain=domain,
        input_data=[
            {"x0": 0.0, "x1": 0.0},
            {"x0": 5.0, "x1": 5.0},
            {"x0": 10.0, "x1": 10.0},
        ],
        output_data=[{"y": -1.0}, {"y": 0.0}, {"y": 1.0}],
    )


def test_lookup_returns_nearest_row():
    pool = _make_pool()
    gen = LookupDataGenerator(pool, input_columns=["x0", "x1"])

    # The candidate (4.4, 4.6) is nearest pool row index 1 (y=0.0).
    sample = ExperimentSample(_input_data={"x0": 4.4, "x1": 4.6})
    result = gen.execute(sample)

    assert result._output_data["y"] == 0.0
    assert str(result.job_status) == "FINISHED"


def test_lookup_corner_distance_picks_corner():
    pool = _make_pool()
    gen = LookupDataGenerator(pool, input_columns=["x0", "x1"])

    sample = ExperimentSample(_input_data={"x0": 9.9, "x1": 9.9})
    result = gen.execute(sample)

    assert result._output_data["y"] == 1.0


def test_lookup_warns_on_repeat():
    pool = _make_pool()
    gen = LookupDataGenerator(pool, input_columns=["x0", "x1"])

    gen.execute(ExperimentSample(_input_data={"x0": 5.0, "x1": 5.0}))
    assert gen.consume_repeats() == 0
    assert gen.seen_indices == {1}

    gen.execute(ExperimentSample(_input_data={"x0": 4.9, "x1": 5.1}))
    assert gen.consume_repeats() == 1
    # Counter resets after consumption.
    assert gen.consume_repeats() == 0


def test_lookup_does_not_mutate_pool():
    pool = _make_pool()
    gen = LookupDataGenerator(pool, input_columns=["x0", "x1"])

    sample = ExperimentSample(_input_data={"x0": 5.0, "x1": 5.0})
    result = gen.execute(sample)

    result._output_data["y"] = 99.0
    # The pool's matched row keeps its original output.
    assert pool.data[1]._output_data["y"] == 0.0


def test_reset_seen_clears_state():
    pool = _make_pool()
    gen = LookupDataGenerator(pool, input_columns=["x0", "x1"])
    gen.execute(ExperimentSample(_input_data={"x0": 0.0, "x1": 0.0}))
    gen.execute(ExperimentSample(_input_data={"x0": 0.0, "x1": 0.1}))
    assert gen.consume_repeats() == 1

    gen.reset_seen()
    assert gen.seen_indices == set()
    gen.execute(ExperimentSample(_input_data={"x0": 0.0, "x1": 0.0}))
    assert gen.consume_repeats() == 0
