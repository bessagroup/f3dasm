"""Extended tests for optimization: factories, imports, error handling."""

import pytest

from f3dasm._src.datageneration.datagenerator_factory import (
    create_datagenerator,
)
from f3dasm._src.optimization._imports import (
    _DeferredImportExceptionContextManager,
    try_import,
)
from f3dasm._src.optimization.errors import faulty_optimizer
from f3dasm.optimization import cg, lbfgsb, nelder_mead

pytestmark = pytest.mark.smoke


# ======================= factory functions =======================


def _dummy_datagenerator():
    return create_datagenerator(
        data_generator="sphere", output_names="y", seed=0
    )


def test_cg_factory():
    opt = cg(
        data_generator=_dummy_datagenerator(),
        output_name="y",
        input_name="x",
    )
    assert opt is not None
    assert opt.method == "CG"


def test_lbfgsb_factory():
    opt = lbfgsb(
        data_generator=_dummy_datagenerator(),
        output_name="y",
        input_name="x",
    )
    assert opt is not None
    assert opt.method == "L-BFGS-B"


def test_nelder_mead_factory():
    opt = nelder_mead(
        data_generator=_dummy_datagenerator(),
        output_name="y",
        input_name="x",
    )
    assert opt is not None
    assert opt.method == "Nelder-Mead"


# ======================= faulty_optimizer =======================


def test_faulty_optimizer_raises():
    with pytest.raises(NotImplementedError):
        faulty_optimizer(name="test_opt", missing_package="test_pkg")


# ======================= DeferredImport =======================


def test_deferred_import_success():
    """Successful import should not raise."""
    ctx = _DeferredImportExceptionContextManager()
    with ctx:
        import math  # noqa: F401
    assert ctx.is_successful()
    # check() should not raise
    ctx.check()


def test_deferred_import_failure():
    """Failed import should defer the exception."""
    ctx = _DeferredImportExceptionContextManager()
    with ctx:
        import nonexistent_module_xyz  # noqa: F401
    assert not ctx.is_successful()
    with pytest.raises(ImportError):
        ctx.check()


def test_try_import_returns_context_manager():
    ctx = try_import()
    assert isinstance(ctx, _DeferredImportExceptionContextManager)
