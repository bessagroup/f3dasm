import numpy as np
import pytest
from f3dasm.sampling.sobolsequence import SobolSequencing
from f3dasm.src.designofexperiments import DoE
from f3dasm.src.space import CategoricalSpace, ContinuousSpace, DiscreteSpace


# def check_two_arrays(a: np.array, b: np.array) -> bool:
#     for i, _ in enumerate(a[:, 0]):
#         for j, _ in enumerate(a[0, :]):
#             if isinstance(a[i, j], float) or isinstance(b[i, j], float):
#                 res = np.allclose(a[i, j], b[i, j], rtol=1e-2)
#             else:
#                 res = a[i, j] == b[i, j]

#             if not res:
#                 return False

#     return True


# def test_correct_sampling():
#     seed = 42

#     # Define the parameters
#     x1 = ContinuousSpace(name="x1", lower_bound=2.4, upper_bound=10.3)
#     x2 = DiscreteSpace(name="x2", lower_bound=5, upper_bound=80)
#     x3 = ContinuousSpace(name="x3", lower_bound=10.0, upper_bound=380.3)
#     x4 = CategoricalSpace(name="x4", categories=["test1", "test2", "test3"])
#     x5 = ContinuousSpace(name="x5", lower_bound=0.6, upper_bound=7.3)

#     # Create the design space
#     space = [x1, x2, x3, x4, x5]
#     design = DoE(space)

#     # Construct sampler
#     sobol_sequencing = SobolSequencing(doe=design, seed=seed)

#     numsamples = 5

#     ground_truth_samples = np.array(
#         [
#             [2.4, 10.0, 0.6, 56.0, "test3"],
#             [6.35, 195.15, 3.95, 19.0, "test2"],
#             [8.325, 102.575, 2.275, 76.0, "test3"],
#             [4.375, 287.725, 5.625, 65.0, "test3"],
#             [5.3625, 148.8625, 4.7875, 25.0, "test3"],
#         ]
#     )
#     samples = sobol_sequencing.get_samples(numsamples=numsamples)
#     # print(samples)
#     print(check_two_arrays(samples, ground_truth_samples))
#     assert samples == pytest.approx(ground_truth_samples)


if __name__ == "__main__":  # pragma: no cover
    pytest.main()
