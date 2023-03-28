#                                                                       Modules
# =============================================================================
# Standard
import json
import os

#                                                          Authorship & Credits
# =============================================================================
__author__ = "Jiaxiang Yi (J.Yi@tudelft.nl)"
__credits__ = ["Jiaxiang Yi"]
__status__ = "Stable"
# =============================================================================
#
# =============================================================================


def create_dir(current_folder: str, dir_name: str) -> str:
    """create new directory
    Parameters
    ----------
    current_folder : str
        current working folder
    dirname : str
        new folder name
    Returns
    -------
    str
        path of created folder
    """

    path = os.path.join(current_folder, dir_name)
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as error:
        print(f"Directory {dir_name} can not be created")

    return path


def write_json(sim_info: dict, file_name: str) -> None:
    """write json file for abaqus
    Parameters
    ----------
    sim_info : dict
        dict that constains
    file_name : str
        file name of json file
    """

    with open(file_name, "w") as fp:
        json.dump(sim_info, fp)


def print_banner(message: str, sign="#", length=50) -> None:
    """print banner
    Parameters
    ----------
    message : str
        string output to the screen
    sign : str, optional
        pattern, by default "#"
    length : int, optional
        length of output, by default 50
    """
    print(sign * length)
    print(
        sign * ((length - len(message) - 2) // 2)
        + " "
        + message
        + " "
        + sign * ((length - len(message) - 2) // 2)
    )
    print(sign * length)


class AssertInputs:
    """class to assert the inputs of abaqus simulation is assigned properly or not"""

    @classmethod
    def is_inputs_proper_defined(
        cls, folder_info: dict, sim_info: dict
    ) -> None:
        cls.is_mwd_in_folder_info(folder_info=folder_info)
        cls.is_script_path_in_folder_info(folder_info=folder_info)
        cls.is_cwd_in_folder_info(folder_info=folder_info)
        cls.is_sim_path_in_folder_info(folder_info=folder_info)
        cls.is_sim_script_in_folder_info(folder_info=folder_info)
        cls.is_post_path_in_folder_info(folder_info=folder_info)
        cls.is_post_script_in_folder_info(folder_info=folder_info)
        cls.is_job_name_in_sim_info(sim_info=sim_info)
        cls.is_platform_in_sim_info(sim_info=sim_info)

    @classmethod
    def is_mwd_in_folder_info(cls, folder_info: dict) -> None:
        assert (
            "main_work_directory" in folder_info.keys()
        ), "main_work_directory should in folder_info dict"

    @classmethod
    def is_script_path_in_folder_info(cls, folder_info: dict) -> None:
        assert (
            "script_path" in folder_info.keys()
        ), "script_path should in folder_info dict"

    @classmethod
    def is_cwd_in_folder_info(cls, folder_info: dict) -> None:
        assert (
            "current_work_directory" in folder_info.keys()
        ), "current_work_directory should in folder_info dict"

    @classmethod
    def is_sim_path_in_folder_info(cls, folder_info: dict) -> None:
        assert (
            "sim_path" in folder_info.keys()
        ), "sim_path should in folder_info dict"

    @classmethod
    def is_sim_script_in_folder_info(cls, folder_info: dict) -> None:
        assert (
            "sim_script" in folder_info.keys()
        ), "sim_script should in folder_info dict"

    @classmethod
    def is_post_path_in_folder_info(cls, folder_info: dict) -> None:
        assert (
            "post_path" in folder_info.keys()
        ), "post_path should in folder_info dict"

    @classmethod
    def is_post_script_in_folder_info(cls, folder_info: dict) -> None:
        assert (
            "post_script" in folder_info.keys()
        ), "post_script should in folder_info dict"

    @classmethod
    def is_job_name_in_sim_info(cls, sim_info: dict) -> None:
        assert (
            "job_name" in sim_info.keys()
        ), "job_name should in folder_info dict"

    @classmethod
    def is_platform_in_sim_info(cls, sim_info: dict) -> None:
        assert (
            "platform" in sim_info.keys()
        ), "platform should in folder_info dict"
