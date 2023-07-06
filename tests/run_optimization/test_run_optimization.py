import pytest

from f3dasm.datageneration.functions import FUNCTIONS_7D
from f3dasm.datageneration.functions.function import Function
from f3dasm.design import Domain
from f3dasm.optimization import OPTIMIZERS, Optimizer
from f3dasm.run_optimization import (OptimizationResult,
                                     run_multiple_realizations)
from f3dasm.sampling import LatinHypercube

# @pytest.mark.smoke
# def test_reproducibility(optimizationresults: OptimizationResult):
#     results_json = optimizationresults.to_json()
#     optimizationresults: OptimizationResult = create_optimizationresult_from_json(results_json)


# @pytest.mark.smoke
# @pytest.mark.parametrize("optimizer", OPTIMIZERS)
# @pytest.mark.parametrize("func", FUNCTIONS_7D)
# def test_o(design_7d: DesignSpace, optimizer: Optimizer, func: Function, number_of_samples: int):

#     samples = LatinHypercube(design=design_7d, seed=42).get_samples(number_of_samples)
#     args = {'optimizer': optimizer(data=samples, seed=42),
#             'function': func(dimensionality=design_7d.get_number_of_input_parameters(),
#                              scale_bounds=design_7d.get_bounds()),
#             'sampler': LatinHypercube(design=design_7d, seed=42),
#             'iterations': 100,
#             'realizations': 2,
#             'number_of_samples': number_of_samples,
#             'parallelization': True,
#             'seed': 42
#             }

#     optimizationresults: OptimizationResult = run_multiple_realizations(**args)


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
