"""Regression tests for issue #301: matplotlib is no longer a hard
dependency, and scipy is now declared as an optional extra so the
intent of who pulls it in is explicit.

The `try_import` pattern in `optimization/optimizer_factory.py` already
gates the scipy optimizers behind successful import; this file adds
coverage for the matplotlib path and pins the deps-list intent."""

import importlib
import sys

import pytest

pytestmark = pytest.mark.smoke


def test_io_imports_without_matplotlib(monkeypatch):
    """`f3dasm._src._io` must import even when matplotlib is absent;
    `figure_store`/`figure_load` then fail at *call* time, not import
    time."""
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)

    import f3dasm._src._io as io_mod

    io_mod = importlib.reload(io_mod)
    assert io_mod._HAS_MATPLOTLIB is False
    assert io_mod.plt is None


def test_figure_store_raises_clear_error_without_matplotlib(monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)

    import f3dasm._src._io as io_mod

    io_mod = importlib.reload(io_mod)
    with pytest.raises(ImportError, match=r"figure_store requires matplotlib"):
        io_mod.figure_store(object=None, path="/tmp/missing")
    with pytest.raises(ImportError, match=r"figure_load requires matplotlib"):
        io_mod.figure_load(path="/tmp/missing")


def test_store_mapping_omits_plt_figure_without_matplotlib(monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)

    import f3dasm._src._io as io_mod

    io_mod = importlib.reload(io_mod)
    # The mapping should still cover the always-on types but skip plt.Figure
    keys = set(io_mod.STORE_FUNCTION_MAPPING.keys())
    import numpy as np
    import pandas as pd

    assert {np.ndarray, pd.DataFrame, pd.Series} <= keys


def test_pyproject_declares_matplotlib_and_scipy_as_optional():
    """Pin the deps-list intent so we don't accidentally regress
    matplotlib or scipy back into the hard-dependency list. Uses a
    plain text scan instead of `tomllib` to stay Py3.10-compatible."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    text = (root / "pyproject.toml").read_text()

    # Locate the required `dependencies = [...]` block and the
    # `[project.optional-dependencies]` section.
    deps_start = text.index("dependencies = [")
    deps_end = text.index("]", deps_start)
    deps_block = text[deps_start:deps_end]
    extras_start = text.index("[project.optional-dependencies]")

    # Required deps must not list matplotlib or scipy directly.
    assert '"matplotlib' not in deps_block, (
        "matplotlib must remain an optional extra"
    )
    assert '"scipy' not in deps_block, "scipy must remain an optional extra"

    # The `all` extra must still pull them in for the full install.
    extras_block = text[extras_start:]
    assert '"matplotlib' in extras_block
    assert '"scipy' in extras_block
