"""
Interface class for data generators
"""

#                                                                       Modules
# =============================================================================

import json
import os
import pickle
import subprocess
from pathlib import Path
from time import perf_counter, sleep
from typing import Any, Dict

from ..logger import logger
from .datagenerator import DataGenerator

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Alpha"
# =============================================================================
#
# =============================================================================


class AbaqusSimulator(DataGenerator):
    EXECUTE_COMMAND = "abaqus cae noGUI=abaqus_script.py -mesa"
    POST_PROCESS_COMMAND = "abaqus cae noGUI=abaqus_post_process.py -mesa"

    def __init__(self, job_name: str = "job", platform: str = "ubuntu", num_cpus: int = 1,
                 script_python_file: str = None, function_name_execute: str = None,
                 script_parent_folder_path: str = None, max_time: float = None,
                 sleep_time: float = 20.0, refresh_time: float = 5.0,
                 delete_odb: bool = True, post_python_file: str = None,
                 function_name_post: str = None, **kwargs):
        """Abaqus simulator class

        Parameters
        ----------
        job_name : str, optional
            Name of the job, by default "job"
        platform : str, optional
            Platform to use; either 'cluster' or 'ubuntu', by default "ubuntu"
        num_cpus : int, optional
            Number of CPUs to use, by default 1
        script_python_file : str, optional
            name of the .py file that needs to be executed by abaqus, by default None
        function_name_execute : str, optional
            Python function or class that is called, by default None
        script_parent_folder_path : str, optional
            parent folder where the script_python_file and script_post_processing are located. By default None
        max_time : float, optional
            (platform=ubuntu only) Number of seconds before killing the simulation, by default None
        sleep_time : float, optional
            (platform=ubuntu only) Number of seconds to wait before checking the log file
            for the first time, by default 20.0
        refresh_time : float, optional
            (platform=ubuntu onlyo) Number of seconds to wait before checking the log file, by default 5.0
        delete_odb : bool, optional
            Set true if you want to delete the original .odb file after post-processing, by default True
        post_python_file : str, optional
            name of the .py file that is needed for post-processing by abaqus, by default None
        function_name_post : str, optional
            name of the .py file that needs to be executed for post-processing by abaqus, by default None


        Notes
        -----
        The kwargs are saved as attributes to the class. This is useful for the
        simulation script to access the parameters.

        The platform is an artifact from the original code. The TU Delft Abaqus
        version is broken, so the process needs to be manually killed.

        The execute function that is called (argument 'function_name_execute'), should be callable
        and accept one dictionary argument. This dictionary contains the parameters
        that are passed to the simulation script.

        The post-processing function that is called (argument 'function_name_execute'), should be callable
        and accept one dictionary argument. This dictionary contains the parameters
        that are passed to the simulation script.

        The post-processing function should read the .odb file and save the results to a results.p file as a dictionary
        The simulator.post_process() function will read this dictionary from disk and stores the arguments to
        the ExperimentSample object.
        """

        # Rule of thumb:
        # All arguments that are specific in the __init__ function are only used
        # for setting up the ABAQUS simulator.
        # All extra keyword arguments are used to pass to the simulation and post-processing scripts!

        # Running parameters
        self.job_name = job_name
        self.max_time = max_time
        self.platform = platform
        self.num_cpus = num_cpus  # TODO: Where do I specify this in the execution of abaqus?
        self.sleep_time = sleep_time
        self.refresh_time = refresh_time
        self.delete_odb = delete_odb

        # Script location parameters
        self.script_parent_folder_path = script_parent_folder_path
        self.script_python_file = script_python_file
        self.function_name_execute = function_name_execute
        self.post_python_file = post_python_file
        self.function_name_post = function_name_post

        # add all arguments to the sim_info dictionary
        self.sim_info = kwargs

        # add the job name to the sim_info dictionary
        self.sim_info["job_name"] = job_name

    def _make_execute_script(self):
        with open("abaqus_script.py", "w") as file:
            file.write("import os\n")
            file.write("import sys\n")
            file.write("import json\n")
            file.write(f"sys.path.extend([r'{self.script_parent_folder_path}'])\n")
            file.write(
                f"from {self.script_python_file} import {self.function_name_execute}\n"
            )
            line = "file = 'sim_info.json'\n"
            file.write(line)
            file.write("with open(file, 'r') as f:\n")
            file.write("    dict = json.load(f)\n")
            file.write(f"{self.function_name_execute}(dict)\n")

    def _make_execute_script_pickle(self):
        with open("abaqus_script.py", "w") as file:
            file.write("import os\n")
            file.write("import sys\n")
            file.write("import pickle\n")
            file.write(f"sys.path.extend([r'{self.script_parent_folder_path}'])\n")
            file.write(
                f"from {self.script_python_file} import {self.function_name_execute}\n"
            )
            line = "file = 'sim_info.pkl'\n"
            file.write(line)
            file.write("with open(file, 'rb') as f:\n")
            file.write("    dict = pickle.load(f)\n")
            file.write(f"{self.function_name_execute}(dict)\n")

    def _make_post_process_script(self):
        with open("abaqus_post_process.py", "w") as file:
            file.write("import os\n")
            file.write("import sys\n")
            file.write(f"sys.path.extend([r'{self.script_parent_folder_path}'])\n")
            file.write(
                f"from {self.post_python_file} import {self.function_name_post}\n"
            )
            file.write(f"{self.function_name_post}('{self.job_name}')\n")

    def execute(self) -> None:

        #############################
        # Creating execution script
        #############################

        # Save cwd for later
        self.home_path: str = os.getcwd()

        # Updating simulation info with experiment sample data
        self.sim_info.update(self.experiment_sample.input_data)
        self.sim_info.update(self.experiment_sample.output_data_loaded)

        # Create working directory
        working_dir = Path(f"case_{self.experiment_sample.job_number}")
        working_dir.mkdir(parents=True, exist_ok=True)

        # Change to working directory
        os.chdir(working_dir)

        # Wrtie sim_info to a JSON file
        # with open("sim_info.json", "w") as fp:
        #     json.dump(self.sim_info, fp)

        with open("sim_info.pkl", "wb") as fp:
            pickle.dump(self.sim_info, fp, protocol=0)

        # Create python file for abaqus to run
        self._make_execute_script_pickle()

        #############################
        # Running Abaqus
        #############################

        # Log start of simulation
        logger.info(f"({self.experiment_sample.job_number}) ABAQUS: {self.script_python_file}")

        if self.platform == 'ubuntu':
            self._run_abaqus_broken_version()
        elif self.platform == 'cluster':
            self._run_abaqus()

        else:
            raise NotImplementedError(f"Platform '{self.platform}' not be implemented"
                                      f"Choose from 'ubuntu' or 'cluster'")

        #############################
        # Post-analysis script
        #############################
        logger.info(f"({self.experiment_sample.job_number}) ABAQUS POST: {self.script_python_file}")

        # path with the post-processing python-script
        self._make_post_process_script()

        os.system(self.POST_PROCESS_COMMAND)

        # remove files that influence the simulation process
        # remove_files(directory=os.getcwd())

        # remove the odb file to save memory
        # if self.delete_odb:
        #     remove_files(directory=os.getcwd(), file_types=[".odb"])

        with open("results.pkl", "rb") as fd:
            results = pickle.load(fd, fix_imports=True, encoding="latin1")

        # Back to home path
        os.chdir(self.home_path)

        # Store results in self.results so that you can access it later
        self.results: Dict[str, Any] = results

    def post_process(self) -> None:
        """Function that handles the post-processing"""

        # for every key in self.results, store tNonehe value in the ExperimentSample object
        for key, value in self.results.items():
            # Check if value is of one of these types: int, float, str, list
            if isinstance(value, (int, float, str)):
                self.experiment_sample.store(object=value, name=key, to_disk=False)

            else:
                self.experiment_sample.store(object=value, name=key, to_disk=True)

    def _run_abaqus(self) -> str:
        start_time = perf_counter()
        os.system(self.EXECUTE_COMMAND)
        end_time = perf_counter()
        logger.info(f"simulation time :{(end_time - start_time):2f} s")

    def _run_abaqus_broken_version(self) -> str:
        proc = subprocess.Popen(self.EXECUTE_COMMAND, shell=True)

        start_time = perf_counter()
        sleep(self.sleep_time)
        while True:

            sleep(
                self.refresh_time - ((perf_counter() - start_time) % self.refresh_time)
            )
            end_time = perf_counter()
            #
            try:
                file = open(f"{self.job_name}.msg")
                word1 = "THE ANALYSIS HAS BEEN COMPLETED"
                if word1 in file.read():
                    proc.kill()
                    kill_abaqus_process()
                    break
            except Exception:
                logger.info(
                    "abaqus license is not enough,"
                    "waiting for license authorization"
                )
            # over time killing
            if self.max_time is not None:
                if (end_time - start_time) > self.max_time:
                    proc.kill()
                    kill_abaqus_process()
                    logger.info("overtime kill")
                    break
        logger.info(f"simulation time :{(end_time - start_time):2f} s")
        # remove files that influence the simulation process
        remove_files(directory=os.getcwd())


def kill_abaqus_process() -> None:
    """kill abaqus simulation process"""
    os.system("pkill standard")
    os.system("pkill ABQcaeK")
    os.system("pkill SMAPython")


def remove_files(
    directory: str, file_types: list = [".log", ".lck", ".SMABulk", ".rec", ".SMAFocus",
                                        ".exception", ".simlog", ".023", ".exception"],
) -> None:
    """Remove files of specified types in a directory.

    Parameters
    ----------
    directory : str
        Target folder.
    file_types : list
        List of file extensions to be removed.
    """
    # Create a Path object for the directory
    dir_path = Path(directory)

    for target_file in file_types:
        # Use glob to find files matching the target extension
        target_files = dir_path.glob(f"*{target_file}")

        # Remove the target files if they exist
        for file in target_files:
            if file.is_file():
                file.unlink()
