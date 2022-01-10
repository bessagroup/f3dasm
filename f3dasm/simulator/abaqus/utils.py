import os
import shutil
import f3dasm
import glob

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


def clean_abaqus_dir(ext2rem=('.abq', '.com', '.log', '.mdl', '.pac', '.rpy',
                              '.sel', '.stt'),
                     dir_path=None):

    # local functions
    def rmfile(ftype):
        file_list = glob.glob(os.path.join(dir_path, '*' + ftype))
        for file in file_list:
            try:
                os.remove(file)
            except:
                pass

    # initialization
    dir_path = os.getcwd() if dir_path is None else dir_path

    # deleting files
    for ftype in ext2rem:
        rmfile(ftype)

    # delete another .rpy
    if '.rpy' in ext2rem:
        ftype = 'rpy.*'
        rmfile(ftype)