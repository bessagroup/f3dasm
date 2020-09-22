'''
Created on 2019-09-12 16:51:02
Last modified on 2020-09-21 12:00:21

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------
Run Abaqus examples that are presented in examples/abaqus.

Notes
-----
-to run in the cmd: >>> abaqus python run_abaqus_example.py <example_name> --gui
-if --gui is not added to cmd, then it runs without gui
-Abaqus work directory will be the one that contains this file (this file needs
to get access to f3das).
'''


# imports

# standard library
import os
import argparse
from itertools import chain

# local library
from f3das.utils.file_handling import verify_existing_name
from f3das.utils.file_handling import clean_abaqus_dir
from f3das.utils.file_handling import get_sorted_by_time


# parse arguments

parser = argparse.ArgumentParser()
parser.add_argument('example_name', type=str,
                    help='Example name. It should be in the folder Abaqus examples')
parser.add_argument('--gui',
                    action='store_true',
                    help='Abaqus cae interface is opened')
parser.add_argument('--same_dir',
                    action='store_true',
                    help='Uses most recent `simulation` dir as simulation directory')

args = parser.parse_args()


# find example in subfolders

folder_names = ['examples']
example_name = os.path.splitext(args.example_name)[0] + '.py'

for path, _, filenames in chain.from_iterable([os.walk(folder_name) for folder_name in folder_names]):
    for filename in filenames:
        if filename.endswith(example_name):
            name = os.path.splitext(os.path.join(path, filename))[0]
            break


# create simul dir
simul_dir_name = 'simulation' if not args.same_dir or not os.path.exists('simulation') else get_sorted_by_time('simulation')[-1]
if not args.same_dir or not os.path.exists(simul_dir_name):
    simul_dir_name = verify_existing_name(simul_dir_name)
    os.mkdir(simul_dir_name)


# generate run file

run_filename = verify_existing_name('_temp.py')
module_name = '.'.join(os.path.normpath(name).split(os.sep))
lines = ['import runpy',
         'import os',
         'import sys',
         'initial_wd = os.getcwd()',
         'sys.path.append(initial_wd)',
         "os.chdir('%s')" % simul_dir_name,
         "runpy.run_module('%s', run_name='__main__')" % module_name,
         'os.chdir(initial_wd)']
with open(run_filename, 'w') as f:
    for line in lines:
        f.write(line + '\n')


# generate example

interface_cmd = 'SCRIPT' if args.gui else 'noGUI'
command = 'abaqus cae ' + interface_cmd + '=%s' % run_filename
fail = os.system(command)


# clean abaqus dir and finalize script

clean_abaqus_dir()
os.remove(run_filename)

if fail:
    raise Exception("'%s' has a bug!" % os.path.splitext(example_name)[0])
else:
    print("'%s' successfully run" % os.path.splitext(example_name)[0])
