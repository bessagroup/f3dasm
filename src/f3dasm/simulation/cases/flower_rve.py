#                                                                       Modules
# =============================================================================
# Standard
import json
import os

import f3dasm
from f3dasm.design import ExperimentData
from f3dasm.simulation.abaqus_simulator import AbaqusSimulator
from f3dasm.simulation.abaqus_utils import create_dir

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Jiaxiang Yi (J.Yi@tudelft.nl)"
__credits__ = ["Jiaxiang Yi"]
__status__ = "Stable"
# =============================================================================
#
# =============================================================================


class FlowerRVE:
    def __init__(self) -> None:
        """Interface between python and abaqus of the Hollowplate case"""

        self.main_folder = os.getcwd()
        self.samples = None
        self.sim_info = None
        self.folder_info = {
            "main_work_directory": os.path.join(os.getcwd(), "Data"),
            "script_path": os.path.dirname(f3dasm.simulation.__file__),
            "current_work_directory": "point_1",
            "sim_path": "abaqus_script.flower_rve_script",
            "sim_script": "FlowerRVE",
            "post_path": "abaqus_script.flower_rve_script",
            "post_script": "FlowerRVEPostProcess",
        }

        self.update_sim_info(print_info=False)

    def update_sim_info(
        self,
        C1: float = 0.2,
        C2: float = 0.1,
        print_info: bool = False,
    ) -> None:

        self.sim_info = {
            "job_name": "flower_rve",
            "MAT_Name": "Arruda",
            "C1": C1,
            "C2": C2,
            "platform": "ubuntu",
        }
        if print_info is True:
            print("Simulation information: \n")
            print(json.dumps(self.sim_info, indent=4))

    def run_simulation(
        self,
        sample: dict = None,
        folder_index: int = None,
        sub_folder_index: int = None,
        third_folder_index: int = None,
    ) -> dict:
        # number of samples
        self._create_working_folder(
            folder_index,
            sub_folder_index,
            third_folder_index,
        )
        # update the geometry info for microstructure
        self._update_sample_info(sample=sample)
        # change folder to main folder
        os.chdir(self.main_folder)
        simulator = AbaqusSimulator(
            sim_info=self.sim_info, folder_info=self.folder_info
        )
        # run abaqus simulation
        simulator.run()
        # get the simulation results back
        results = simulator.read_back_results()

        return results

    def run_batch_simulation(self) -> any:
        pass

    def run_f3dasm(self, data: ExperimentData):

        # get the samples
        samples = data.data.input.to_dict("record")
        for ii in range(len(data.data)):
            results = self.run_simulation(
                sample=samples[ii], third_folder_index=ii
            )
            # fill the data class
            data.data["output"] = data.data["output"].astype(object)
            for jj in range(len(list(data.data["output"].keys()))):
                data.data[("output", list(data.data["output"].keys())[jj])][
                    ii
                ] = results[list(data.data["output"].keys())[jj]]
        # data.to_json()
        return data

    def _update_sample_info(self, sample) -> None:
        """update the design variables"""
        self.sim_info.update(sample)

    def _create_working_folder(
        self,
        folder_index: float = None,
        sub_folder_index: float = None,
        third_folder_index: float = None,
    ) -> None:
        """_summary_

        Parameters
        ----------
        folder_index : float, optional
            first order, by default None
        sub_folder_index : float, optional
            sub-folder, by default None
        third_folder_index : float, optional
            sub-sub-folder, by default None

        Raises
        ------
        ValueError
            folder index error
        ValueError
            folder index error
        """
        if folder_index is None:
            if sub_folder_index is None:
                self.folder_info["current_work_directory"] = "case_" + str(
                    third_folder_index
                )
            else:
                if third_folder_index is None:
                    self.folder_info[
                        "current_work_directory"
                    ] = "point_" + str(sub_folder_index)
                else:
                    self.folder_info["current_work_directory"] = (
                        "point_"
                        + str(sub_folder_index)
                        + "/case_"
                        + str(third_folder_index)
                    )
        else:
            if sub_folder_index is None:
                raise ValueError("provide sub_folder_index")
            elif third_folder_index is None:
                raise ValueError("provide third_folder_index")
            else:
                self.folder_info["current_work_directory"] = (
                    "gen_"
                    + str(folder_index)
                    + "/point_"
                    + str(sub_folder_index)
                    + "/case_"
                    + str(third_folder_index)
                )
        new_path = create_dir(
            current_folder=self.folder_info["main_work_directory"],
            dir_name=self.folder_info["current_work_directory"],
        )
        self.working_folder = new_path
        os.chdir(new_path)
