'''
Created on 2019-09-12 16:51:02
Last modified on 2020-12-16 18:48:10

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
Run Abaqus examples.
'''


# imports

# standard library
import os
import argparse
import shutil
from pathlib import Path

# local library
from ..utils.file_handling import verify_existing_name
from ..utils.file_handling import clean_abaqus_dir
from ..utils.file_handling import get_sorted_by_time
from ..run.abaqus import _create_temp_dir


# function definition

def get_args():

    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('example_name', type=str,
                        help='Example name. It should be in the folder Abaqus examples')
    parser.add_argument('--dir', type=str, default='.',
                        help='Directory where to look for <example_name>. Default: current dir')
    parser.add_argument('--gui',
                        action='store_true',
                        help='Abaqus cae interface is opened')
    parser.add_argument('--same_dir',
                        action='store_true',
                        help='Uses most recent `simulation` dir as simulation directory')

    args = parser.parse_args()

    # get
    example_name = '{}.py'.format(os.path.splitext(args.example_name)[0])

    return args.dir, example_name, args.same_dir, args.gui


def run_example(folder_name, example_name, same_dir, gui):

    # find example in subfolders
    module_name = _find_module_name(folder_name, example_name)

    # create simul dir
    simul_dir_name = _create_simul_dir('simulation', same_dir)

    # create __init__.py files
    files_to_del = _create_init_files(module_name)

    # copy f3dasm if not in the current directory
    if 'f3dasm' not in os.listdir('.'):
        temp_dir_name = _create_temp_dir('_temp')
    else:
        temp_dir_name = ''

    # generate run file
    run_filename = _generate_run_file(module_name, simul_dir_name, temp_dir_name,
                                      'temp.py')

    # run example
    interface_cmd = 'SCRIPT' if gui else 'noGUI'
    command = 'abaqus cae ' + interface_cmd + '=%s' % run_filename
    fail = os.system(command)

    # clean dir
    _clean_dir(run_filename, temp_dir_name, files_to_del)

    # success message
    if fail:
        raise Exception("'%s' has a bug!" % os.path.splitext(example_name)[0])
    else:
        print("'%s' successfully run" % os.path.splitext(example_name)[0])


def _find_module_name(folder_name, example_name):
    module_name = None
    for path, _, filenames in os.walk(folder_name):
        for filename in filenames:
            if filename.lower().endswith(example_name.lower()):
                module_name = os.path.splitext(os.path.join(path, filename))[0]
                break

    if module_name is None:
        raise Exception('{} was not found'.format(example_name))

    return module_name


def _create_simul_dir(name, same_dir):
    simul_dir_name = name if not same_dir or not os.path.exists(name) else get_sorted_by_time(name)[-1]
    if not same_dir or not os.path.exists(simul_dir_name):
        simul_dir_name = verify_existing_name(simul_dir_name)
        os.mkdir(simul_dir_name)

    return simul_dir_name


def _create_init_files(module_name):

    p = Path(module_name)
    path = ''
    init_filename = '__init__.py'
    files_to_del = []
    for part in p.parts[:-1]:
        path = os.path.join(path, part)
        if init_filename not in os.listdir(path):
            filename_ = os.path.join(path, init_filename)
            # create empty file
            open(filename_, 'w').close()
            files_to_del.append(filename_)

    return files_to_del


def _generate_run_file(module_name, simul_dir_name, temp_dir_name, temp_filename):

    run_filename = verify_existing_name(temp_filename)
    module_name = '.'.join(os.path.normpath(module_name).split(os.sep))
    lines = ['import runpy',
             'import os',
             'import sys',
             'initial_wd = os.getcwd()',
             "temp_dir_name = '%s'" % temp_dir_name,
             'sys.path.append(initial_wd)',
             'if temp_dir_name:',
             '\tsys.path.append(os.path.join(initial_wd, temp_dir_name))',
             "os.chdir('%s')" % simul_dir_name,
             "runpy.run_module('%s', run_name='__main__')" % module_name,
             'os.chdir(initial_wd)']
    with open(run_filename, 'w') as f:
        for line in lines:
            f.write(line + '\n')

    return run_filename


def _clean_dir(run_filename, temp_dir_name, files_to_del):

    clean_abaqus_dir()
    os.remove(run_filename)
    for filename in files_to_del:
        os.remove(filename)
        try:
            os.remove('{}c'.format(filename))
        except FileNotFoundError:
            pass
    if temp_dir_name:
        shutil.rmtree(temp_dir_name)


if __name__ == '__main__':

    # get arguments
    folder_name, example_name, same_dir, gui = get_args()

    # run example
    run_example(folder_name, example_name, same_dir, gui)
