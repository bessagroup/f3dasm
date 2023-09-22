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
    EXECUTE_COMMAND = "abaqus cae noGUI=" + "abaScript.py" + " -mesa"
    POST_PROCESS_COMMAND = "abaqus cae noGUI=" + "getResults.py" + " -mesa"

    def __init__(self, platform: str = "ubuntu",
                 sim_path: str = None, sim_script: str = None, script_path: str = None, max_time: float = None,
                 sleep_time: float = 20.0, refresh_time: float = 5.0,
                 delete_odb: bool = True, post_path: str = None,
                 post_script: str = None, **kwargs):
        """Abaqus simulator class

        Parameters
        ----------
        platform : str, optional
            Platform to use; either 'cluster' or 'ubuntu', by default "ubuntu"
        sim_path : str, optional
            name of the .py file that needs to be executed by abaqus, by default None
        sim_script : str, optional
            Python function or class that is called, by default None
        script_path : str, optional
            parent folder where the sim_path and post_path are located. By default None
        max_time : float, optional
            (platform=abaqus only) Number of seconds before killing the simulation, by default None
        sleep_time : float, optional
            (platform=abaqus only) Number of seconds to wait before checking the log file
            for the first time, by default 20.0
        refresh_time : float, optional
            (platform=abaqus onlyo) Number of seconds to wait before checking the log file, by default 5.0
        delete_odb : bool, optional
            Set true if you want to delete the original .odb file after post-processing, by default True
        post_path : str, optional
            parent folder of where the simulation post-processing script is located, by default None
        post_script : str, optional
            name of the .py file that needs to be executed for post-processing by abaqus, by default None


        Notes
        -----
        The kwargs are saved as attributes to the class. This is useful for the
        simulation script to access the parameters.

        The platform is an artifact from the original code. The TU Delft Abaqus
        version is broken, so the process needs to be manually killed.

        The class or function that is called (argument 'sim_script'), should be callable
        and accept one dictionary argument. This dictionary contains the parameters
        that are passed to the simulation script.
        """
        self.max_time = max_time
        self.platform = platform
        self.sim_path = sim_path
        self.script_path = script_path
        self.sim_script = sim_script
        self.post_path = post_path
        self.post_script = post_script
        self.sleep_time = sleep_time
        self.refresh_time = refresh_time
        self.delete_odb = delete_odb

        self.sim_info = kwargs

        # save kwargs to attributes
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _make_execute_script(self):
        with open("abaScript.py", "w") as file:
            file.write("import os \n")
            file.write("import sys \n")
            file.write("import json \n")
            file.write(
                "sys.path.extend(['"
                + str(self.script_path)
                + "']) \n"
            )
            file.write(
                "from "
                + str(self.sim_path)
                + " import "
                + str(self.sim_script)
                + "\n"
            )
            line = "file = '" + "sim_info.json" + "' \n"
            file.write(line)
            file.write("with open(file, 'r') as f:\n")
            file.write("	dict = json.load(f)\n")
            file.write(str(self.sim_script) + "(dict)\n")
        file.close()

    def _make_post_process_script(self, file_name: str, script_path: str, post_path: str, post_script: str):
        with open("getResults.py", "w") as file:
            file.write("import os\n")
            file.write("import sys\n")
            file.write(
                "sys.path.extend(['"
                + str(self.script_path)
                + "']) \n"
            )
            file.write(
                "from "
                + str(self.post_path)
                + " import "
                + str(self.post_script)
                + "\n"
            )
            file.write(
                str(self.post_script)
                + "('"
                + str("job")
                + "')\n"
            )
        file.close()

    def execute(self) -> None:

        #############################
        # Creating execution script
        #############################

        # Save cwd for later
        self.home_path: str = os.getcwd()

        # Updating simulation info with experiment sample data
        self.sim_info.update(self.experiment_sample.input_data)
        self.sim_info.update(self.experiment_sample.output_data)

        # Create working directory
        working_dir = Path(f"case_{self.experiment_sample.job_number}")
        working_dir.mkdir(parents=True, exist_ok=True)

        # Change to working directory
        os.chdir(working_dir)

        # Wrtie sim_info to a JSON file
        with open("sim_info.json", "w") as fp:
            json.dump(self.sim_info, fp)

        # Create python file for abaqus to run
        self._make_execute_script()

        #############################
        # Running Abaqus
        #############################

        # Log start of simulation
        logger.info("Start ABAQUS Simulation")

        if self.platform == 'ubuntu':
            self._run_abaqus_broken_version()
        elif self.platform == 'cluster':
            self._run_abaqus()

        else:
            raise NotImplementedError("platform not be implemented")

        #############################
        # Post-analysis script
        #############################

        logger.info("abaqus post analysis")

        # path with the post-processing python-script
        self._make_post_process_script()

        os.system(self.POST_PROCESS_COMMAND)

        # remove files that influence the simulation process
        remove_files(directory=os.getcwd())

        # remove the odb file to save memory
        if self.delete_odb:
            remove_files(directory=os.getcwd(), file_types=[".odb"])

        with open("results.p", "rb") as fd:
            results = pickle.load(fd, fix_imports=True, encoding="latin1")

        # Back to home path
        os.chdir(self.home_path)

        # Store results in self.results so that you can access it later
        self.results = results

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
                file = open("job" + ".msg")
                word1 = "THE ANALYSIS HAS BEEN COMPLETED"
                if word1 in file.read():
                    proc.kill()
                    kill_abaqus_process()
                    break
            except Exception:
                print(
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
    directory: str,
    file_types: list = [
        ".log",
        ".lck",
        ".SMABulk",
        ".rec",
        ".SMAFocus",
        ".exception",
        ".simlog",
        ".023",
        ".exception",
    ],
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

    # Get all files in this folder
    all_files = dir_path.iterdir()

    for target_file in file_types:
        # Get the target file names
        filtered_files = [file for file in all_files if file.name.endswith(target_file)]

        # Remove the target files if they exist
        for file in filtered_files:
            if file.is_file():
                file.unlink()
