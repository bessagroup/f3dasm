import os
import shutil


def quickstart():
    """
    Quickstart a study
    Copy files and directories from a subdirectory of the current module to the current working directory.
    """
    source_dir = 'files'
    file_list = ['config.py', 'config.yaml', 'default.yaml', 'main.py', 'pbsjob.sh', 'README.md', 'hydra']

    module_path = os.path.dirname(os.path.abspath(__file__))
    source_path = os.path.join(module_path, source_dir)
    for file in file_list:
        source_file = os.path.join(source_path, file)
        dest_file = os.path.join(os.getcwd(), file)
        if os.path.isdir(source_file):
            shutil.copytree(source_file, dest_file)
        else:
            shutil.copyfile(source_file, dest_file)
