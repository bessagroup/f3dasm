import pytest

from f3dasm.run_optimization import (OptimizationResult,
                                     create_optimizationresult_from_json)


def test_reproducibility(optimizationresults: OptimizationResult):
    results_json = optimizationresults.to_json()
    create_optimizationresult_from_json(results_json)


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
