import f3dasm
import numpy as np


def main():
    # Define the blocks:
    dimensionality = 20
    iterations = 20
    realizations = 10
    seed = 42

    hyperparameters = {}  # If none are selected, the default ones are used

    domain = np.tile([-1.0, 1.0], (dimensionality, 1))
    design = f3dasm.make_nd_continuous_design(bounds=domain, dimensionality=dimensionality)
    data = f3dasm.Data(design)

    implementation = {
        "realizations": realizations,
        "optimizer": f3dasm.optimization.PSO(data=data),
        "function": f3dasm.functions.Levy(dimensionality=dimensionality, noise=False, scale_bounds=domain),
        "sampler": f3dasm.sampling.SobolSequenceSampling(design, seed=seed),
        "iterations": iterations,
    }

    results = f3dasm.run_multiple_realizations(**implementation)


if __name__ == "__main__":
    main()
