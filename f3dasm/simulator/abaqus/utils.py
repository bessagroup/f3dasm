import os
import shutil
import f3dasm


def create_temp_dir(temp_dir_name='_temp'):
    if not os.path.exists(temp_dir_name):
        os.mkdir(temp_dir_name)
    new_f3das_dir = os.path.join(temp_dir_name, 'f3dasm')
    if os.path.exists(new_f3das_dir):
        shutil.rmtree(new_f3das_dir)
    shutil.copytree(f3dasm.__path__[0], new_f3das_dir)



def run_job_from_inp(inp_file, sim_dir):
    initial_wd  = os.getcwd()
    os.chdir(sim_dir)
    command = 'abaqus job={}'.format(inp_file)
    os.system(command)
    os.chdir(initial_wd)


