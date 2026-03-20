"""Extended tests for optimization: factories, imports, error handling."""

import pytest

from f3dasm._src.optimization._imports import (
    _DeferredImportExceptionContextManager,
    try_import,
)
from f3dasm._src.optimization.errors import faulty_optimizer
from f3dasm._src.optimization.optimizer_factory import create_optimizer

pytestmark = pytest.mark.smoke


# ======================= create_optimizer =======================


def test_create_optimizer_invalid_name():
    with pytest.raises(KeyError):
        create_optimizer(optimizer="nonexistent_optimizer")


def test_create_optimizer_invalid_type():
    with pytest.raises(TypeError):
        create_optimizer(optimizer=12345)


def test_create_optimizer_cg():
    opt = create_optimizer(optimizer="cg")
    assert opt is not None


def test_create_optimizer_lbfgsb():
    opt = create_optimizer(optimizer="lbfgsb")
    assert opt is not None


def test_create_optimizer_neldermead():
    opt = create_optimizer(optimizer="neldermead")
    assert opt is not None


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
