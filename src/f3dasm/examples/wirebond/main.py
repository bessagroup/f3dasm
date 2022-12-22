import logging
from typing import Any, List
import autograd.numpy as np
import hydra
from hydra.core.config_store import ConfigStore
import f3dasm
from config import Config
import os
import shutil
import fileinput

class Wirebond_function(f3dasm.Function):

    def __init__(
        self, 
        dimensionality: int = 2, 
        seed: Any or int = None,
        fidelity_value: float = None,
        ):
        super().__init__(dimensionality, seed)

        self.fidelity_value = fidelity_value

    def f(self, x: np.ndarray):

        res = []

        for i, xi in enumerate(x):

            WThk = round(0.39 * xi[0] + 0.1, 8)
            FL = round(0.7 * xi[1] + 0.5, 8)

            MeshSize_XY = round(2 - 0.9 * self.fidelity_value, 8)
        
            work_folder_name = 'wirebond_multifidelity'
            out_path = work_folder_name + '/Max_Strain_WThk=' + str(WThk) \
                + '_FL=' +  str(FL) \
                + '_MeshSizeXY=' + str(MeshSize_XY) + '.txt'

            geometric_inputs = work_folder_name + '/01_Geometric_Inputs.txt'
            mesh_parameters = work_folder_name + '/02_Mesh_Parameters.txt'

            if not os.path.exists(work_folder_name):
                # os.mkdir(work_folder_name)
                shutil.copytree(
                    "/home/leoguo/Documents/GitHub/F3DASM/src/f3dasm/examples/wirebond/resources/wirebond_multifidelity_source",
                    # "/home/leoguo/wirebondBO/F3DASM/src/f3dasm/examples/wirebond/resources/wirebond_multifidelity_source",
                     work_folder_name
                     )

            geometric_inputs_file = fileinput.FileInput(files=geometric_inputs, inplace=True)
            for line in geometric_inputs_file:
                if "WThk = " in line: # Wire thickness
                    print('WThk = ' + str(WThk), end='\n')
                elif "FL = " in line: # Foot length
                    print('FL = ' + str(FL), end='\n')
                else:
                    print(line, end='')

            mesh_parameters_file = fileinput.FileInput(files=mesh_parameters, inplace=True)
            for line in mesh_parameters_file:
                if "MeshSize_XY = " in line: # XY-plane mesh size
                    print('MeshSize_XY = ' + str(MeshSize_XY), end='\n')
                else:
                    print(line, end='')

            cmdl = '"ansys2019r3" -p ansys -dis -mpi INTELMPI -np 2 -lch -dir "' + work_folder_name + '" -j "wirebond_ms_' \
                + str(MeshSize_XY) + '" -s read -l en-us -b -i "' + work_folder_name + '/01_Geometric_Inputs.txt"'
            # os.system(cmdl)

            if os.path.exists(out_path):
                out_content = open(out_path, 'rb').read()
                resi = np.array(float(out_content))
            else:
                # resi = np.random.rand()
                resi = np.nan
                f = open(out_path,"w")
                f.write(str(resi))

                raise 'nan value'
            
            res.append(resi)

        return np.array(res).reshape(-1, 1)

def convert_config_to_input(config: Config) -> List[dict]:

    # seed = np.random.randint(low=0, high=1e5)
    seed = config.execution.seed

    optimizer_class: f3dasm.Optimizer = f3dasm.find_class(f3dasm.optimization, config.optimizer.optimizer_name)

    sampler_class: f3dasm.SamplingInterface = f3dasm.find_class(f3dasm.sampling, config.sampler.sampler_name)

    bounds = np.tile([config.design.lower_bound, config.design.upper_bound], (config.design.dimensionality, 1))
    design = f3dasm.make_nd_continuous_design(bounds=bounds, dimensionality=config.design.dimensionality)
    data = f3dasm.Data(design=design)

    optimizer = optimizer_class(data=data, seed=seed)
    optimizer.init_parameters()
    optimizer.parameter.noise_fix = False

    sampler = sampler_class(design=data.design, seed=seed)

    fidelity_functions = []
    multifidelity_samplers = []

    for i, fidelity_value in enumerate(config.function.fidelity_values):

        fun = Wirebond_function(fidelity_value=fidelity_value)
        
        parameter_DesignSpace = f3dasm.make_nd_continuous_design(
            bounds=np.tile([0.0, 1.0], (config.design.dimensionality, 1)),
            dimensionality=config.design.dimensionality,
        )
        fidelity_parameter = f3dasm.ConstantParameter(name="fid", constant_value=fidelity_value)
        parameter_DesignSpace.add_input_space(fidelity_parameter)

        sampler = f3dasm.sampling.SobolSequence(design=parameter_DesignSpace)

        fidelity_functions.append(fun)

        multifidelity_samplers.append(sampler)

        multifidelity_function = f3dasm.MultiFidelityFunction(
            fidelity_functions=fidelity_functions,
            fidelity_parameters=config.function.fidelity_values,
            costs=config.function.costs,
)

    return {
        "optimizer": optimizer,
        "multifidelity_function": multifidelity_function,
        "multifidelity_samplers": multifidelity_samplers,
        "numbers_of_samples": config.sampler.numbers_of_samples,
        "iterations": config.execution.iterations,
        "seed": seed,
        "budget": config.execution.budget,
        }


@hydra.main(config_path=".", config_name="default")
def main(cfg: Config):
    options = convert_config_to_input(config=cfg)

    # options['optimizer'].init_parameters()
    # options['optimizer'].parameter.noise_fix = False

    result = f3dasm.run_multi_fidelity_optimization(**options)
    result[-1].data.to_csv('testrun.csv')

    print(result[-1].data)

cs = ConfigStore.instance()
cs.store(name="f3dasm_config", node=Config)

log = logging.getLogger(__name__)

if __name__ == "__main__":
    main()
