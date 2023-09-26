"""
This file contains the functions that are used to run Abaqus simulations.

Note
----
I'm assuming that every process could have a different pre-processing, execution and post-processing function.
But they are used in that order and only once.
If multiple e.g. pre_process steps need to be taken, then they should be combined in one function.   
"""

#                                                                       Modules
# =============================================================================

# Standard
import os
import pickle
from functools import partial
from pathlib import Path
from typing import Any, Dict

# Local
from ...design.experimentsample import ExperimentSample

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Alpha"
# =============================================================================
#
# =============================================================================


def pre_process(experiment_sample: ExperimentSample, folder_path: str,
                python_file: str, function_name: str = "main", **kwargs) -> None:
    """Function that handles the pre-processing of Abaqus with a Python script

    Parameters
    ----------
    experiment_sample : ExperimentSample
        The design to run the data generator on. Will be handled by the pipeline.
    folder_path : str
        Path of the folder where the python script is located
    python_file : str
        Name of the python file to be executed
    function_name : str, optional
        Name of the function within the python file to be executed, by default "main"

    Note
    ----
    The python file should create an .inp input-file based on the information of the
    experiment sample named <job_number>.inp.
    """

    sim_info = kwargs

    # Updating simulation info with experiment sample data
    sim_info.update(experiment_sample.to_dict())

    filename = f"sim_info_{experiment_sample.job_number}.pkl"
    with open(filename, "wb") as fp:
        pickle.dump(sim_info, fp, protocol=0)

    with open("preprocess.py", "w") as file:
        file.write("import os\n")
        file.write("import sys\n")
        file.write("import pickle\n")
        file.write(f"sys.path.extend([r'{folder_path}'])\n")
        file.write(
            f"from {python_file} import {function_name}\n"
        )
        file.write(f"with open('{filename}', 'rb') as f:\n")
        file.write("    dict = pickle.load(f)\n")
        file.write(f"{function_name}(dict)\n")

    os.system("abaqus cae noGUI=preprocess.py -mesa")


def post_process(experiment_sample: ExperimentSample, folder_path: str,
                 python_file: str, function_name: str = "main") -> None:
    """Function that handles the post-processing of Abaqus with a Python script

    Parameters
    ----------
    experiment_sample : ExperimentSample
        The design to run the data generator on. Will be handled by the pipeline.
    folder_path : str
        Path of the folder where the python script is located
    python_file : str
        Name of the python file to be executed
    function_name : str, optional
        Name of the function within the python file to be executed, by default "main"

    Note
    ----
    The post-processing python file should write the results of your simulation to a pickle file
    with the name: results.pkl. This file will be handled by the pipeline.
    """

    with open("post.py", "w") as file:
        file.write("import os\n")
        file.write("import sys\n")
        file.write("from abaqus import session\n")
        file.write(f"sys.path.extend([r'{folder_path}'])\n")
        file.write(
            f"from {python_file} import {function_name}\n"
        )
        file.write(f"odb = session.openOdb(name='{experiment_sample.job_number}.odb')\n")
        file.write(f"{function_name}(odb)\n")

    os.system("abaqus cae noGUI=post.py -mesa")
