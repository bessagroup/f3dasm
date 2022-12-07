# import system packages
import os
import sys
import time
import subprocess
import pickle
import numpy as np

# import local functions
from .abaqus_utils import (
    make_dir,
    write_json,
    kill_abaqus_processes,
    print_banner,
)

from ...base.simulation import Simulator


class AbaqusSimulator(Simulator):
    """ """

    def __init__(self, sim_info: dict, folder_info: dict):
        """
        Parameters
        ----------
        sim_info:
        folder_info
        """
        self.sim_info = sim_info
        self.folder_info = folder_info
        self.job_name = sim_info["job_name"]
        self.main_work_directory = folder_info["main_work_directory"]
        self.current_work_directory = folder_info["current_work_directory"]
        self.script_path = folder_info["script_path"]
        self.sim_path = folder_info["sim_path"]
        self.post_path = folder_info["post_path"]

        self.sim_info_name = "sim_info.json"

    def execute(self, x) -> None:
        """
        Parameters
        ----------
        sim_info: contain the information that used for the abaqus simulation
        folder_info: folder operation so that the needed information is restored in corresponding folders
        test_data; the data used to test
        Returns
        -------
        """

        self.set_input_parameters(x)

        # hidden name information
        new_python_filename = "abqScript.py"

        # folder operations
        make_dir(current_folder=self.main_work_directory,
                 dirname=self.current_work_directory)

        # change work directory
        os.chdir(os.path.join(self.main_work_directory,
                 self.current_work_directory))

        print(f"Current working directory: {os.getcwd()}")

        # output the sim_info dict to a json file
        write_json(sim_info=self.sim_info, filename=self.sim_info_name)

        self.make_new_script(
            new_file_name=new_python_filename,
            script_path=self.folder_info["sim_path"],
        )

        # begin to run abaqus simulation and submit the job to get the .odb file
        print_banner("START ABAQUS ANALYSIS")

        self._run_abaqus_sim(new_python_filename)
        self._remove_files()

        os.chdir(self.main_work_directory)

    def post_process(self):
        os.chdir(os.path.join(self.main_work_directory,
                 self.current_work_directory))

        print_banner("START ABAQUS POST PROCESSING")
        # path with the python-script

        post_process_python_filename = "get_results.py"

        self.make_new_script(
            new_file_name=post_process_python_filename,
            script_path=self.folder_info["post_path"],
        )

        self._run_abaqus(post_process_python_filename)
        self._remove_files()

        # change the work directory back the main one
        os.chdir(self.main_work_directory)

    def set_input_parameters(self, x: np.ndarray) -> None:
        # Input paramaters
        self.sim_info["C1"] = x[0]
        self.sim_info["C2"] = x[1]

    def read_back_results(self):
        """

        Returns
        -------

        """
        os.chdir(os.path.join(self.main_work_directory,
                 self.current_work_directory))

        with open("results.p", "rb") as fd:
            results = pickle.load(fd, fix_imports=True, encoding="latin1")

        os.chdir(self.main_work_directory)

        return results

    def make_new_script(
        self,
        new_file_name: str,
        script_path: str,
    ) -> None:

        with open(new_file_name, "w") as file:
            file.write("import os \n")
            file.write("import sys \n")
            file.write("import json \n")
            file.write(
                "sys.path.extend(['" + self.folder_info["script_path"] + "']) \n")
            file.write("from " + script_path + " import main" + "\n")
            file.write("file = '" + self.sim_info_name + "' \n")
            file.write("with open(file, 'r') as f:\n")
            file.write("	input_dict = json.load(f)\n")
            file.write("main(input_dict)\n")
        file.close()

    # --------- run the abaqus python file -----------------

    def _run_abaqus(self, python_filename: str):
        proc = subprocess.Popen("abaqus cae noGUI=" +
                                python_filename + " -mesa", shell=True)
        return proc

    def _run_abaqus_sim(self, python_filename: str):
        """
        This function is used to run abaqus simulation
        Returns:  do not return any thing but need to teminate the simulation process
        The reason is the abaqus software has some problems in this desktop
        -------
        """
        # my_cmd_job = command
        proc = self._run_abaqus(python_filename)
        # proc = subprocess.Popen("abaqus cae noGUI=" + python_filename + " -mesa", shell=True)
        start_time = time.time()

        while True:
            # in this part: check the job is finish or not !
            #  check the word in msg file existed or not (THE ANALYSIS HAS BEEN COMPLETED)
            time.sleep(20.0 - ((time.time() - start_time) % 20.0))
            msg_name = self.job_name + ".msg"
            file = open(msg_name)
            word1 = "THE ANALYSIS HAS BEEN COMPLETED"
            word2 = "ABAQUS/standard rank 0 terminated by signal 11 (Segmentation fault)"
            if word1 in file.read():
                print("Simulation successfully finished! \n")
                proc.kill()
                kill_abaqus_processes()
                break
            elif word2 in file.read():
                print("Simulation failed due to error! \n")
                proc.kill()
                kill_abaqus_processes()
                break

        end_time = time.time()
        print(f"the simulation time is :{end_time - start_time} !")

    def _remove_file(self, directory, file_type: str):
        files_in_directory = os.listdir(directory)
        filtered_files = [
            file for file in files_in_directory if file.endswith(file_type)]
        for file in filtered_files:
            path_to_file = os.path.join(directory, file)
            if os.path.exists(path_to_file):
                os.remove(path_to_file)

    def _remove_files(self):
        log_file = self.job_name + ".log"
        if os.path.exists(log_file):
            os.remove(log_file)
        lck_file = self.job_name + ".lck"
        if os.path.exists(lck_file):
            os.remove(lck_file)
        directory = os.getcwd()

        list_of_removed_filetypes = [
            ".SMABulk", ".rec", ".SMAFocus", ".exception", ".simlog"]

        for filetype in list_of_removed_filetypes:
            self._remove_file(directory, filetype)
