import os
import pickle
# I'm assuming that every process could have a different pre-processing, execution and post-processing function.
# But they are used in that order and only once.
# If multiple e.g. pre_process steps need to be taken, then they should be combined in one function.
from functools import partial
from pathlib import Path
from typing import Any, Dict

from ..design.experimentsample import ExperimentSample


def pre_process(experiment_sample: ExperimentSample, script_parent_folder_path: str,
                script_python_file: str, function_name_execute: str = "main", **kwargs) -> None:

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
        file.write(f"sys.path.extend([r'{script_parent_folder_path}'])\n")
        file.write(
            f"from {script_python_file} import {function_name_execute}\n"
        )
        file.write(f"with open('{filename}', 'rb') as f:\n")
        file.write("    dict = pickle.load(f)\n")
        file.write(f"{function_name_execute}(dict)\n")

    os.system("abaqus cae noGUI=preprocess.py -mesa")
    # This will create an .inp file with the name: f"{experiment_sample.job_number}.inp"


def post_process(experiment_sample: ExperimentSample, script_parent_folder_path: str,
                 post_python_file: str, function_name_post: str = "main") -> None:
    """Function that handles the post-processing"""

    with open("post.py", "w") as file:
        file.write("import os\n")
        file.write("import sys\n")
        file.write("from abaqus import session\n")
        file.write(f"sys.path.extend([r'{script_parent_folder_path}'])\n")
        file.write(
            f"from {post_python_file} import {function_name_post}\n"
        )
        file.write(f"odb = session.openOdb(name='{experiment_sample.job_number}.odb')\n")
        file.write(f"{function_name_post}(odb)\n")

    os.system("abaqus cae noGUI=post.py -mesa")
    # This will create a .pkl file with the name: f"results_{experiment_sample.job_number}.pkl"
