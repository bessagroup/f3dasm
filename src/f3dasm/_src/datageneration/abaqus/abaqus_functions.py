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
from pathlib import Path

# Local
from ...experimentdata.experimentsample import ExperimentSample

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Martin van der Schelling (M.P.vanderSchelling@tudelft.nl)"
__credits__ = ["Martin van der Schelling"]
__status__ = "Alpha"
# =============================================================================
#
# =============================================================================


def pre_process(experiment_sample: ExperimentSample, folder_path: str,
                python_file: str, function_name: str = "main", name: str = "job",
                remove_temp_files: bool = True, **kwargs) -> None:
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
    name : str, optional
        Name of the job, by default "job"
    remove_temp_files : bool, optional

    Note
    ----
    The python file should create an .inp input-file based on the information of the
    experiment sample named <job_number>.inp.
    """

    sim_info = kwargs
    working_dir = Path("datageneration") / Path(f"{name}_{experiment_sample.job_number}")

    # Updating simulation info with experiment sample data
    sim_info.update(experiment_sample.to_dict())

    filename = working_dir / "sim_info.pkl"
    with open(filename, "wb") as fp:
        pickle.dump(sim_info, fp, protocol=0)

    with open(f"{working_dir / 'preprocess.py'}", "w") as f:
        f.write("import os\n")
        f.write("import sys\n")
        f.write("import pickle\n")
        f.write(f"sys.path.extend([r'{folder_path}'])\n")
        f.write(
            f"from {python_file} import {function_name}\n"
        )
        f.write(f"with open(r'{filename}', 'rb') as f:\n")
        f.write("    dict = pickle.load(f)\n")
        f.write(f"os.chdir(r'{working_dir}')\n")
        f.write(f"{function_name}(dict)\n")

    os.system(f"abaqus cae noGUI={working_dir / 'preprocess.py'} -mesa")

    if remove_temp_files:
        Path(working_dir / "preprocess.py").unlink(missing_ok=True)
        Path(working_dir / "sim_info.pkl").unlink(missing_ok=True)


def post_process(experiment_sample: ExperimentSample, folder_path: str,
                 python_file: str, function_name: str = "main", name: str = "job",
                 remove_temp_files: bool = True, **kwargs) -> None:
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
    name : str, optional
        Name of the job, by default "job"
    remove_temp_files : bool, optional
        Whether to remove the temporary files, by default True

    Note
    ----
    The post-processing python file should write the results of your simulation to a pickle file
    with the name: results.pkl. This file will be handled by the pipeline.
    """

    working_dir = Path("datageneration") / Path(f"{name}_{experiment_sample.job_number}")

    with open(f"{working_dir / 'post.py'}", "w") as f:
        f.write("import os\n")
        f.write("import sys\n")
        f.write("from abaqus import session\n")
        f.write(f"sys.path.extend([r'{folder_path}'])\n")
        f.write(
            f"from {python_file} import {function_name}\n"
        )
        f.write(f"odb = session.openOdb(name=r'{working_dir / str(experiment_sample.job_number)}.odb')\n")
        f.write(f"os.chdir(r'{working_dir}')\n")
        f.write(f"{function_name}(odb)\n")

    os.system(f"abaqus cae noGUI={working_dir / 'post.py'} -mesa")
    if remove_temp_files:
        Path(working_dir / "post.py").unlink(missing_ok=True)
