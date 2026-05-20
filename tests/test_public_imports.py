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


def test_import_pipeline():
    from f3dasm.pipeline import (  # noqa: F401
        CollectArrayResults,
        Loop,
        Pipeline,
        SlurmCluster,
        SlurmResources,
        Step,
    )


def test_import_pipeline_run_step():
    # Module is invoked as ``python -m f3dasm.pipeline.run_step``
    # by SLURM jobs, so it must import cleanly and expose ``main``.
    from f3dasm.pipeline.run_step import main  # noqa: F401


def test_import_pipeline_count_open():
    # Module is invoked as ``python -m f3dasm.pipeline.count_open``
    # by the SLURM orchestrator to size array submissions.
    from f3dasm.pipeline.count_open import main  # noqa: F401
