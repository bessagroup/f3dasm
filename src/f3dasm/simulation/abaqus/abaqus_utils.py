import os
import json


def make_dir(current_folder: str, dirname: str) -> None:
    """
    Make directories
    Parameters
    ----------
    current_folder (str): cwd
    dirname (str): folder that needs to be created
    """

    path = os.path.join(current_folder, dirname)
    try:
        os.makedirs(path, exist_ok=True)

    except OSError as error:
        print(f"Directory {dirname} can not be created")

    print(f"Directory {dirname} created successfully")


def write_json(sim_info: dict, filename: str) -> None:
    """

    Parameters
    ----------
    sim_info: a dict that contains the information for simulation
    filename: a string of the new .py file

    Returns
    -------

    """
    with open(filename, "w") as fp:
        json.dump(sim_info, fp)


def make_new_script(
    new_file_name: str,
    folder_path: str,
    script_path: str,
    sim_info_name: str = "sim_info.json",
) -> None:

    with open(new_file_name, "w") as file:
        file.write("import os \n")
        file.write("import sys \n")
        file.write("import json \n")
        file.write("sys.path.extend(['" + folder_path + "']) \n")
        file.write("from " + script_path + " import main" + "\n")
        file.write("file = '" + str(sim_info_name) + "' \n")
        file.write("with open(file, 'r') as f:\n")
        file.write("	input_dict = json.load(f)\n")
        file.write("main(input_dict)\n")
    file.close()


def kill_abaqus_processes():
    # os.kill(name, signal.SIGKILL)
    name_1 = "pkill standard"
    aa = os.system(name_1)
    name_2 = "pkill ABQcaeK"
    bb = os.system(name_2)
    name_3 = "pkill SMAPython"
    cc = os.system(name_3)
    print(aa + bb + cc)


def print_banner(message: str, sign: str = "#", length: int = 50, suspend: bool = True) -> None:
    if not suspend:
        print(sign * (length))
        print(
            sign * ((length - len(message) - 2) // 2) + " " +
            message + " " + sign * ((length - len(message) - 2) // 2)
        )
        print(sign * (length))
    return
