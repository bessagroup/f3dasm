"""Regression tests for issue #285: LoadFunction must accept **kwargs and
the ``load_kwargs`` registered on the Parameter must be forwarded on every
load.

The flagship use case is loaders like ``equinox.tree_deserialise_leaves``
that need a template (`like=template`) to reconstruct the object."""

from pathlib import Path
from typing import Any

import pytest

from f3dasm import ExperimentData, ExperimentSample
from f3dasm._src._io import (
    ReferenceValue,
    ToDiskValue,
    load_object,
    pickle_load,
)
from f3dasm._src.design.parameter import Parameter
from f3dasm.design import Domain

pytestmark = pytest.mark.smoke


class Skeleton:
    """Stand-in for an `equinox.Module` template used to drive
    deserialisation. Holds the schema the loader needs to reconstruct
    the object."""

    def __init__(self, key: str) -> None:
        self.key = key


def _store_pair(obj: dict, path: str) -> str:
    """Write ``obj`` as a `.pkl` file alongside the chosen extension."""
    import pickle as _pickle

    _path = Path(path).with_suffix(".pkl")
    with open(_path, "wb") as f:
        _pickle.dump(obj, f)
    return str(_path)


def _load_pair(path: str, *, like: Skeleton) -> dict:
    """Load and attach the skeleton key to the deserialised payload.

    Models the equinox case: the loader receives auxiliary state via
    kwargs that is required to interpret the bytes on disk."""
    payload = pickle_load(path)
    return {"payload": payload, "schema_key": like.key}


def test_load_kwargs_forwarded_through_reference_value(tmp_path):
    """`ReferenceValue.load` must forward the registered `load_kwargs`
    to the load function. Without this plumbing the user would have to
    bake `like=...` into a closure and lose the JSON-roundtrip ability
    that f3dasm relies on for `Domain.from_file`."""
    skeleton = Skeleton(key="version-1")
    obj = ToDiskValue(
        object={"value": 42},
        name="custom",
        store_function=_store_pair,
        load_function=_load_pair,
        load_kwargs={"like": skeleton},
    )
    # Set up an experiment-data subfolder so `load_object` resolves paths
    # the same way it does in production.
    project_dir = tmp_path
    (project_dir / "experiment_data").mkdir(parents=True, exist_ok=True)
    reference_path = Path(obj.store(project_dir=project_dir, idx=0))

    ref = obj.to_reference(reference=reference_path)
    loaded = ref.load(project_dir=project_dir)
    assert loaded == {"payload": {"value": 42}, "schema_key": "version-1"}


def test_parameter_rejects_load_kwargs_when_not_to_disk():
    """Setting `load_kwargs` only makes sense when the parameter is
    `to_disk=True`. Mismatch should raise the same way as
    `load_function`."""
    with pytest.raises(ValueError, match="load_kwargs"):
        Parameter(to_disk=False, load_kwargs={"like": object()})


def test_parameter_copy_preserves_load_kwargs():
    """`_copy()` carries `load_kwargs` so domain copies do not silently
    drop the deserialisation hint."""
    skeleton = Skeleton(key="version-2")
    param = Parameter(
        to_disk=True,
        store_function=lambda obj, p: p,
        load_function=lambda p, **kw: kw,
        load_kwargs={"like": skeleton},
    )
    copy = param._copy()
    assert copy.load_kwargs == {"like": skeleton}


def test_load_object_ignores_kwargs_for_builtin_loaders(tmp_path):
    """The default loaders take only `path` -- if no `load_kwargs` are
    registered we must keep calling them with a single positional
    argument so the existing built-ins keep working."""
    import numpy as np

    project_dir = tmp_path
    (project_dir / "experiment_data").mkdir(parents=True, exist_ok=True)
    rel_path = Path("test/0.npy")
    full = project_dir / "experiment_data" / rel_path
    full.parent.mkdir(parents=True, exist_ok=True)
    np.save(full, np.array([1, 2, 3]))

    loaded = load_object(project_dir=project_dir, path=rel_path)
    np.testing.assert_array_equal(loaded, np.array([1, 2, 3]))


def test_load_kwargs_end_to_end_through_experiment_data(tmp_path):
    """End-to-end: domain output declared with `load_kwargs`, sample
    stored, then loaded back -- the auxiliary state must reach the
    custom loader."""
    skeleton = Skeleton(key="version-3")
    domain = Domain()
    domain.add_float(name="x", low=0.0, high=1.0)
    domain.add_output(
        name="custom",
        to_disk=True,
        store_function=_store_pair,
        load_function=_load_pair,
        load_kwargs={"like": skeleton},
    )

    ed = ExperimentData(domain=domain, input_data=[{"x": 0.5}])
    # Use the property setter (not `set_project_dir(..., in_place=True)`)
    # so each ExperimentSample's `project_dir` is propagated -- _store
    # uses `experiment_sample.project_dir` for the on-disk location.
    ed.project_dir = tmp_path

    sample = ed.data[0]
    sample.store(
        name="custom",
        object={"value": 7},
        to_disk=True,
        store_function=_store_pair,
        load_function=_load_pair,
        load_kwargs={"like": skeleton},
    )
    from f3dasm._src.experimentdata import _store as _store_helper

    sample, domain = _store_helper(
        experiment_sample=sample, idx=0, domain=ed.domain
    )

    assert isinstance(sample._output_data["custom"], ReferenceValue)
    loaded = sample._output_data["custom"].load(project_dir=tmp_path)
    assert loaded == {"payload": {"value": 7}, "schema_key": "version-3"}
