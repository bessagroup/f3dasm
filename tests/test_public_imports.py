"""Smoke tests for public re-export wrapper modules."""

import pytest

pytestmark = pytest.mark.smoke


def test_import_datageneration_functions():
    from f3dasm.datageneration.functions import (  # noqa: F401
        ackley,
        booth,
        branin,
        levy,
        rastrigin,
        rosenbrock,
        sphere,
    )


def test_import_hydra_tools():
    from f3dasm.hydra_tools import (  # noqa: F401
        update_config_with_experiment_sample,
    )


def test_import_optimization_scipy():
    from f3dasm.optimization import cg, lbfgsb, nelder_mead  # noqa: F401


def test_import_optimization_optuna():
    optuna = pytest.importorskip("optuna")  # noqa: F841
    from f3dasm.optimization import tpesampler  # noqa: F401
