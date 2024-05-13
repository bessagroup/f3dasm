# import numpy as np
# import pytest

# from f3dasm._src.experimentdata.samplers import RandomUniform, randomuniform

# pytestmark = pytest.mark.smoke


# def test_correct_discrete_sampling_1(design):
#     seed = 42

#     numsamples = 5
#     # Construct sampler
#     random_uniform = randomuniform(
#         domain=design, seed=seed, n_samples=numsamples)

#     random_uniform = random_uniform.data.to_numpy()

#     ground_truth_samples = np.array(
#         [
#             ["test3", "material1"],
#             ["test1", "material3"],
#             ["test3", "material2"],
#             ["test3", "material3"],
#             ["test1", "material3"],
#         ]
#     )
#     # samples = random_uniform._sample_categorical(numsamples=numsamples)

#     assert random_uniform == ground_truth_samples


# def test_correct_discrete_sampling_2(design2):
#     seed = 42

#     # Construct sampler
#     random_uniform = RandomUniform(domain=design2, seed=seed)

#     numsamples = 5

#     ground_truth_samples = np.array(
#         [
#             ["main", "test51", "material6"],
#             ["main", "test14", "material18"],
#             ["main", "test71", "material10"],
#             ["main", "test60", "material10"],
#             ["main", "test20", "material3"],
#         ]
#     )
#     samples = random_uniform._sample_categorical(numsamples=numsamples)

#     assert (samples == ground_truth_samples).all()


# if __name__ == "__main__":  # pragma: no cover
#     pytest.main()
