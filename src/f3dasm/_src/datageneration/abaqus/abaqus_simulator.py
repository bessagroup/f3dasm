"""
Abaqus simulator class
"""

#                                                                       Modules
# =============================================================================

# Standard

import os
import pickle
from pathlib import Path
from typing import Any, Dict

# Local
from ...logger import logger
from ..datagenerator import DataGenerator
from .utils import remove_files

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
                 delete_odb: bool = False, delete_temp_files: bool = True):
        """Abaqus simulator class

        Parameters
        ----------
        name : str, optional
            Name of the job, by default "job"
        num_cpus : int, optional
            Number of CPUs to use, by default 1
        delete_odb : bool, optional
            Set true if you want to delete the original .odb file after post-processing, by default True
        delete_temp_files : bool, optional
            Set true if you want to delete the temporary files after post-processing, by default True

        Notes
        -----
        The kwargs are saved as attributes to the class. This is useful for the
        simulation script to access the parameters.
        """
        self.name = name
        self.num_cpus = num_cpus  # TODO: Where do I specify this in the execution of abaqus?
        self.delete_odb = delete_odb
        self.delete_temp_files = delete_temp_files

    def _pre_simulation(self, **kwargs) -> None:
        """Setting up the abaqus simulator
        - Create working directory: /<name>_<job_number>
        - Changing to this working directory
        """
        # Save cwd for later
        # self.home_path: str = os.getcwd()

        # Create working directory
        self.working_dir = Path("datageneration") / Path(f"{self.name}_{self.experiment_sample.job_number}")
        self.working_dir.mkdir(parents=True, exist_ok=True)

        # Change to working directory
        # os.chdir(working_dir)  # TODO: Get rid of this cwd change

    def execute(self) -> None:
        """Submit the .inp file to run the ABAQUS simulator, creating an .odb file

        Note
        ----
        This will execute the simulation and create an .odb file with name: <job_number>.odb
        """
        filename = self.working_dir / "execute.py"
        logger.info(f"Executing ABAQUS simulator '{self.name}' for sample: {self.experiment_sample.job_number}")

        with open(f"{filename}", "w") as f:
            f.write("from abaqus import mdb\n")
            f.write("import os\n")
            f.write("from abaqusConstants import OFF\n")
            f.write(f"os.chdir(r'{self.working_dir}')\n")
            f.write(
                f"modelJob = mdb.JobFromInputFile(inputFileName="
                f"r'{self.experiment_sample.job_number}.inp',"
                f"name='{self.experiment_sample.job_number}',"
                f"numCpus={self.num_cpus})\n")
            f.write("modelJob.submit(consistencyChecking=OFF)\n")
            f.write("modelJob.waitForCompletion()\n")

        # mdb.jobs['Simul_SUPERCOMPRESSIBLE_LIN_BUCKLE'].setValues(numGPUs=2,
        #     numThreadsPerMpiProcess=1)

        os.system(f"abaqus cae noGUI={filename} -mesa")
        # filename.unlink(missing_ok=True)
        # os.system("abaqus j=input_file.inp cpus=4 ask_delete=OFF")

    def _post_simulation(self):
        """Opening the results.pkl file and storing the data to the ExperimentData object

        Note
        ----
        Temporary files will be removed. If the simulator has its argument 'delete_odb' to True,
        the .odb file will be removed as well to save storage space.

        After the post-processing, the working directory will be changed back to directory
        before running the simulator.

        Raises
        ------
        FileNotFoundError
            When results.pkl is not found in the working directory
        """
        # remove files that influence the simulation process
        # remove_files(directory=Path(), file_types=[".rpy", ".rpy.1", ".rpy.2"])
        # Path("abaqus_acis.log").unlink(missing_ok=True)

        if self.delete_temp_files:
            remove_files(directory=self.working_dir)

        # remove the odb file to save memory
        if self.delete_odb:
            remove_files(directory=self.working_dir, file_types=[".odb"])

        # Check if path exists
        if not Path(self.working_dir / "results.pkl").exists():
            raise FileNotFoundError(f"{Path(self.working_dir) / 'results.pkl'}")

        # Load the results
        with open(Path(self.working_dir / "results.pkl"), "rb") as fd:
            results: Dict[str, Any] = pickle.load(fd, fix_imports=True, encoding="latin1")

        # Back to home path
        # os.chdir(self.home_path)

        # for every key in self.results, store the value in the ExperimentSample object
        for key, value in results.items():
            # Check if value is of one of these types: int, float, str
            if isinstance(value, (int, float, str)):
                self.experiment_sample.store(object=value, name=key, to_disk=False)

            else:
                self.experiment_sample.store(object=value, name=key, to_disk=True)

        # Remove the results.pkl file
        Path(self.working_dir / "results.pkl").unlink(missing_ok=True)
