'''
Created on 2020-04-22 19:50:46
Last modified on 2020-04-28 17:04:02
Python 2.7.16
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
Create auxiliar functions.
'''


#%% imports

# standard library
import os
from collections import OrderedDict
import subprocess
import time

# local library
from ...misc.file_handling import verify_existing_name


#%% run abaqus

def run_simuls_sequentially(example_name, simuls_dir_name, points, wait_time=0,
                            run_module_name='f3das.abaqus.run.run_model',
                            open_shell=True):

    # initialization
    time.sleep(wait_time)

    # create run filename
    run_filename = verify_existing_name('_temp.py')
    lines = ['import runpy',
             'import os',
             'import sys',
             'initial_wd = os.getcwd()',
             'sys.path.append(initial_wd)',
             'points = %s' % points,
             "sim_dir = r'%s'" % os.path.join(example_name, simuls_dir_name),
             'for point in points:',
             "\tos.chdir('%s' % os.path.join(sim_dir, 'DoE_point%i' % point))",
             "\trunpy.run_module('%s', run_name='__main__')" % run_module_name,
             '\tos.chdir(initial_wd)']
    with open(run_filename, 'w') as f:
        for line in lines:
            f.write(line + '\n')

    # open abaqus and run module
    command = 'abaqus cae noGUI=%s ' % run_filename
    # subprocess.check_output(command, shell=open_shell)
    os.system(command)

    # clear direction
    os.remove(run_filename)


def get_missing_simuls(example_name, simuls_dir_name, simul_pkl_name='simul'):

    missing_simuls = []
    dir_path = os.path.join(example_name, simuls_dir_name)
    for folder_name in os.listdir(dir_path):
        folder_path = os.path.join(dir_path, folder_name)
        if os.path.isdir(folder_path) and ('%s.pkl_abaqus' % simul_pkl_name) not in os.listdir(folder_path):
            missing_simuls.append(int(folder_name[9:]))

    return sorted(missing_simuls)


#%% function definition

def convert_dict_unicode_str(pickled_dict):
    new_dict = OrderedDict() if type(pickled_dict) is OrderedDict else {}
    for key, value in pickled_dict.items():
        value = _set_converter_flow(value)
        new_dict[str(key)] = value

    return new_dict


def convert_iterable_unicode_str(iterable):
    new_iterable = []
    for value in iterable:
        value = _set_converter_flow(value)
        new_iterable.append(value)

    if type(iterable) is tuple:
        new_iterable = tuple(new_iterable)
    elif type(iterable) is set:
        new_iterable = set(new_iterable)

    return new_iterable


def _set_converter_flow(value):

    if type(value) is unicode:
        value = str(value)
    elif type(value) in [OrderedDict, dict]:
        value = convert_dict_unicode_str(value)
    elif type(value) in [list, tuple, set]:
        value = convert_iterable_unicode_str(iter)

    return value
