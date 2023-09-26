"""
Interface class for data generators
"""

#                                                                       Modules
# =============================================================================

import json
import os
import pickle
import subprocess
from copy import copy
from pathlib import Path
from time import perf_counter, sleep
from typing import Any, Dict

from ...logger import logger
from ..datagenerator import DataGenerator

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Alpha"
# =============================================================================
#
# =============================================================================


class AbaqusSimulator(DataGenerator):
    def __init__(self, name: str = "job", num_cpus: int = 1,
                 delete_odb: bool = True, **kwargs):
        """Abaqus simulator class

        Parameters
        ----------
        name : str, optional
            Name of the job, by default "job"
        num_cpus : int, optional
            Number of CPUs to use, by default 1
        delete_odb : bool, optional
            Set true if you want to delete the original .odb file after post-processing, by default True

        Notes
        -----
        The kwargs are saved as attributes to the class. This is useful for the
        simulation script to access the parameters.
        """
        self.name = name
        self.num_cpus = num_cpus  # TODO: Where do I specify this in the execution of abaqus?
        self.delete_odb = delete_odb

    def _pre_simulation(self) -> None:
        # Save cwd for later
        self.home_path: str = os.getcwd()

        # Create working directory
        working_dir = Path(f"{self.name}_{self.experiment_sample.job_number}")
        working_dir.mkdir(parents=True, exist_ok=True)

        # Change to working directory
        os.chdir(working_dir)  # TODO: Get rid of this cwd change

    def execute(self) -> None:
        with open("execute.py", "w") as file:
            file.write("from abaqus import mdb\n")
            file.write("from abaqusConstants import OFF\n")
            file.write(
                f"modelJob = mdb.JobFromInputFile(inputFileName='{self.experiment_sample.job_number}.inp',"
                f"name='{self.experiment_sample.job_number}')\n")
            file.write("modelJob.submit(consistencyChecking=OFF)\n")
            file.write("modelJob.waitForCompletion()\n")

        os.system("abaqus cae noGUI=execute.py -mesa")
        # This will execute the simulation and create an .odb file with name: f"{experiment_sample.job_number}.odb"

    def _post_simulation(self):

        # remove files that influence the simulation process
        remove_files(directory=os.getcwd())

        # remove the odb file to save memory
        if self.delete_odb:
            remove_files(directory=os.getcwd(), file_types=[".odb"])

        # Check if path exists
        if not Path("results.pkl").exists():
            raise FileNotFoundError("results.pkl")

        # Load the results
        with open("results.pkl", "rb") as fd:
            results: Dict[str, Any] = pickle.load(fd, fix_imports=True, encoding="latin1")

        # Back to home path
        os.chdir(self.home_path)

        # for every key in self.results, store the value in the ExperimentSample object
        for key, value in results.items():
            # Check if value is of one of these types: int, float, str
            if isinstance(value, (int, float, str)):
                self.experiment_sample.store(object=value, name=key, to_disk=False)

            else:
                self.experiment_sample.store(object=value, name=key, to_disk=True)


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
