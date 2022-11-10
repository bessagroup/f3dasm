import os

import numpy as np

import f3dasm


def main():

    # NEW F3DASM

    N = 1  # number of samples
    dimensionality = 2
    bounds = [0.05, 0.3]  # box-constrained boundaries

    space = f3dasm.make_nd_continuous_design(np.tile([bounds[0], bounds[1]], (dimensionality, 1)), dimensionality)
    data = f3dasm.Data(design=space)

    sampler = f3dasm.sampling.LatinHypercube(design=space)
    sampler.get_samples(N)

    data = f3dasm.Data(design=space)
    data.add(sampler.get_samples(N).data)

    x = data.get_input_data().iloc[0].to_numpy()  # 1.99419533e-01, 7.24177347e-02

    # ABAQUS IMPLEMENTATION

    path_abaqus_script = os.path.dirname(f3dasm.simulation.abaqus.__file__)

    sim_info = {"MAT_Name": "Arruda", "job_name": "Job-1"}  # "C1": C1, "C2": C2,
    # "C1": 1.99419533e-01, "C2": 7.24177347e-02, "MAT_Name": "Neohookean"

    wd_path = os.path.join(os.getcwd(), "data")
    folder_info = {
        "main_work_directory": wd_path,
        "script_path": path_abaqus_script,
        "current_work_directory": "point_1",
        "sim_path": "ExampleFlower.FlowerRVESim",
        "post_path": "ExampleFlower.FlowerRVEPostProcess",
    }

    abaqus_wrapper = f3dasm.simulation.abaqus.AbaqusSimulator(sim_info=sim_info, folder_info=folder_info)
    abaqus_wrapper.run(x)
    results = abaqus_wrapper.read_back_results()

    return results


if __name__ == "__main__":
    main()
