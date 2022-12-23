import fileinput
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import shutil

import f3dasm

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

        MeshSize_XY = round(2 - 0.9 * self.fidelity_value, 6)

        res = []
        history_file = 'src/f3dasm/examples/wirebond/sweep_MeshSize=' + str(MeshSize_XY) + '.csv'

        for xi in x:

            WThk = round(0.39 * xi[0] + 0.1, 6)
            FL = round(0.7 * xi[1] + 0.5, 6)


            if not os.path.exists(history_file):
                df = pd.DataFrame(columns=['WThk', 'FL', 'MeshSize_XY', 'MaxEPS'])
                df.to_csv(history_file, index=False)
            
            df = pd.read_csv(history_file)
            df_update = pd.DataFrame([[WThk, FL, MeshSize_XY, np.nan]], columns=['WThk', 'FL', 'MeshSize_XY', 'MaxEPS'])
            df = pd.concat([df, df_update])
            df.to_csv(history_file, index=False)
        
            work_folder_name = 'wirebond_multifidelity'
            out_path = work_folder_name + '/Max_Strain_WThk=' + str(WThk) \
                + '_FL=' +  str(FL) \
                + '_MeshSizeXY=' + str(MeshSize_XY) + '.txt'

            geometric_inputs = work_folder_name + '/01_Geometric_Inputs.txt'
            mesh_parameters = work_folder_name + '/02_Mesh_Parameters.txt'

            if not os.path.exists(work_folder_name):
                # os.mkdir(work_folder_name)
                shutil.copytree(
                    # "/home/leoguo/Documents/GitHub/F3DASM/src/f3dasm/examples/wirebond/resources/wirebond_multifidelity_source",
                    # "/home/leoguo/wirebondBO/F3DASM/src/f3dasm/examples/wirebond/resources/wirebond_multifidelity_source",
                    "src/f3dasm/examples/wirebond/resources/wirebond_multifidelity_source",
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
            os.system(cmdl)

            if os.path.exists(out_path):
                out_content = open(out_path, 'rb').read()
                resi = np.array(float(out_content))
            else:
                # resi = np.random.rand()
                resi = np.nan
                # f = open(out_path, "w")
                # f.write(str(resi))

                # raise 'nan value'

            df.iloc[-1]['MaxEPS'] = resi
            df.to_csv(history_file, index=False)
            
            res.append(resi)

            shutil.rmtree(work_folder_name)
            
        return np.array(res).reshape(-1, 1)

fun = Wirebond_function(fidelity_value=0.5)

parameter_DesignSpace: f3dasm.DesignSpace = f3dasm.make_nd_continuous_design(
    bounds=np.tile([0.0, 1.0], (fun.dimensionality, 1)),
    dimensionality=fun.dimensionality,
)

sampler = f3dasm.sampling.SobolSequence(design=parameter_DesignSpace)

train_data: f3dasm.Data = sampler.get_samples(numsamples=200)

train_data.add_output(output=fun(train_data))