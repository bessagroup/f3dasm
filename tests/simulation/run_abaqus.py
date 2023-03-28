#                                                                       Modules
# =============================================================================

# Standard
import os
import sys

# Third-party
import numpy as np

# Locals
folder_path = "/home/jiaxiangyi/Documents/F3DASM/src"
sys.path.insert(0, folder_path)

import f3dasm
from f3dasm.design import ExperimentData
from f3dasm.simulation.cases.flower_rve import FlowerRVE

#                                                          Authorship & Credits
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling", "Jiaxiang Yi"]
__status__ = "Stable"
# =============================================================================
#
# =============================================================================


def main() -> ExperimentData:

    # NEW F3DASM

    N = 2  # number of samples

    # define the doe
    C1 = f3dasm.ContinuousParameter(
        name="C1", lower_bound=0.05, upper_bound=0.3
    )
    C2 = f3dasm.ContinuousParameter(
        name="C2", lower_bound=0.05, upper_bound=0.3
    )
    # define the output
    stress = f3dasm.ContinuousParameter(name="stress")
    strain = f3dasm.ContinuousParameter(name="strain")

    design = f3dasm.DesignSpace(
        input_space=[C1, C2], output_space=[stress, strain]
    )

    sampler = f3dasm.sampling.LatinHypercube(design=design, seed=1)
    data = sampler.get_samples(numsamples=N)

    # initialize the simulation
    problem = FlowerRVE()
    problem.update_sim_info(print_info=True)

    # run simulation by import doe
    # results = {}
    # for ii in range(N):
    #     results[ii] = problem.run_simulation(
    #         sample=samples[ii], third_folder_index=ii
    #     )

    results = problem.run_f3dasm(data=data)
    print(results.data)
    return results


if __name__ == "__main__":
    main()
